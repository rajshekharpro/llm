# generation/eval_generation.py
import os
import json
import math
import torch
import torch.nn.functional as F
from models.transformer import DecoderOnlyTransformer
from word_tokenizer.tokenizer import WordTokenizer   # adjust if folder name differs
from utils.data_utils import download_tinystories
from generation.generate import generate_sample
from tqdm import tqdm
import evaluate
import csv

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(path="data/tokenizer.json"):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tok = WordTokenizer()
    tok.word2idx = data["word2idx"]
    # build idx2word correctly
    # If keys are strings of words, reverse mapping:
    if isinstance(list(tok.word2idx.keys())[0], str):
        tok.idx2word = {v: k for k, v in tok.word2idx.items()}
    else:
        # fallback
        tok.idx2word = {int(i): w for i, w in enumerate(tok.word2idx.keys())}
    return tok

def load_model_from_checkpoint(checkpoint_path, tokenizer, config):
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        context_len=config["context_len"],
        embedding_weights=None,
        freeze_embeddings=False
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    if "model_state" in ckpt:
        state = ckpt["model_state"]
    else:
        state = ckpt
    model.load_state_dict(state)
    model.eval()
    return model

def encode_prompt_without_eos(tokenizer, text):
    # Build prompt token ids with <sos> and tokens of text (without adding eos)
    toks = text.strip().split()
    ids = [tokenizer.word2idx.get(tokenizer.sos_token)]
    for t in toks:
        ids.append(tokenizer.word2idx.get(t, tokenizer.word2idx.get(tokenizer.unk_token)))
    return ids

def score_generated_sequence(model, tokenizer, prompt_ids, generated_ids):
    """
    Compute total negative log-likelihood (NLL) of the generated tokens under the model.
    Returns (nll_sum, num_generated_tokens)
    """
    # ids_for_scoring = prompt + generated (keep eos if present)
    ids_for_scoring = prompt_ids + generated_ids
    if len(ids_for_scoring) < 2:
        return 0.0, 0
    ids_tensor = torch.tensor([ids_for_scoring], dtype=torch.long, device=DEVICE)  # (1, T)
    # logits for inputs ids[:-1]
    logits, _, _ = model(ids_tensor[:, :-1])  # (1, T-1, V)
    targets = ids_tensor[:, 1:]               # (1, T-1)
    B, Lm1, V = logits.shape
    logits_flat = logits.view(-1, V)          # ((T-1), V)
    targets_flat = targets.view(-1)           # ((T-1),)

    # mask positions that correspond to generated tokens only
    # target at position i (0-based) corresponds to full-sequence token at index i+1
    # generated tokens are those with full-sequence index >= len(prompt_ids)
    mask = torch.zeros_like(targets_flat, dtype=torch.bool, device=DEVICE)
    for i in range(targets_flat.size(0)):
        full_token_index = i + 1
        if full_token_index >= len(prompt_ids):
            mask[i] = True

    if mask.sum().item() == 0:
        return 0.0, 0

    # compute per-position NLL
    nll_all = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # shape (T-1,)
    nll_generated = nll_all[mask]
    nll_sum = nll_generated.sum().item()
    n_tokens = int(mask.sum().item())
    return nll_sum, n_tokens

def detokenize_from_ids(tokenizer, ids):
    return tokenizer.decode(ids)

def evaluate_many(model, tokenizer, checkpoint_info, num_samples=50, max_new_tokens=50, temperature=1.0, top_k=40, sample=True, out_csv="generation_eval.csv"):
    # Load dataset texts and take the validation split similar to training split logic
    texts = download_tinystories()
    split = int(0.9 * len(texts))
    val_texts = texts[split: split + 5000]  # take a subset
    val_texts = val_texts[:num_samples]

    bleu = evaluate.load("bleu")

    total_nll = 0.0
    total_tokens = 0
    hyps = []
    refs = []
    rows = []

    for idx, full_text in enumerate(tqdm(val_texts, desc="Eval samples")):
        # build prompt = first 5 tokens (or fewer)
        words = full_text.strip().split()
        prompt_words = words[:5] if len(words) >= 5 else words
        prompt_text = " ".join(prompt_words)
        prompt_ids = encode_prompt_without_eos(tokenizer, prompt_text)  # list

        # generate using model.generate (returns full sequence including prompt ids)
        with torch.no_grad():
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
            generated_tensor = model.generate(input_tensor, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, sample=sample, tokenizer=tokenizer, device=DEVICE)
            gen_ids_list = generated_tensor[0].tolist()

        # compute continuation ids (tokens beyond prompt in returned seq)
        # find start index: prompt length
        start_idx = len(prompt_ids)
        continuation_ids = gen_ids_list[start_idx:]
        # Remove trailing PADs if any, and handle EOS (stop at first eos)
        # map eos id if available
        eos_id = tokenizer.word2idx.get(tokenizer.eos_token)
        cleaned_cont = []
        for tok in continuation_ids:
            if tok == eos_id:
                break
            cleaned_cont.append(tok)

        if len(cleaned_cont) == 0:
            # nothing generated, skip scoring
            # but still add empty hyp/reference
            hyp_text = ""
            ref_words = words[len(prompt_words):]  # ground truth continuation
            ref_text = " ".join(ref_words)
            hyps.append([])
            refs.append([ref_words])
            rows.append((prompt_text, ref_text, hyp_text, float("nan")))
            continue

        # Score NLL over generated tokens
        nll_sum, n_tokens = score_generated_sequence(model, tokenizer, prompt_ids, cleaned_cont)
        total_nll += nll_sum
        total_tokens += n_tokens

        # Detokenize hypothesis and reference continuation (without prompt)
        hyp_text = detokenize_from_ids(tokenizer, cleaned_cont)
        ref_words = words[len(prompt_words):]
        ref_text = " ".join(ref_words)

        # Prepare for BLEU: BLEU expects token lists
        hyps.append(hyp_text.split())
        refs.append([ref_words])

        # per-sample perplexity
        per_sample_ppl = math.exp(nll_sum / n_tokens) if n_tokens > 0 else float("nan")
        rows.append((prompt_text, ref_text, hyp_text, per_sample_ppl))

    # compute final metrics
    avg_perplexity = math.exp(total_nll / total_tokens) if total_tokens > 0 else float("nan")
    bleu_res = bleu.compute(predictions=[" ".join(h) for h in hyps], references=[[ " ".join(r[0]) ] for r in refs])
    avg_bleu = bleu_res.get("bleu", float("nan"))

    # save CSV
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt", "reference_continuation", "hypothesis_continuation", "per_sample_ppl"])
        for r in rows:
            writer.writerow(r)

    return {"avg_perplexity": avg_perplexity, "avg_bleu": avg_bleu, "total_generated_tokens": total_tokens, "n_samples": len(rows)}

if __name__ == "__main__":
    # EDIT THESE PATHS / CONFIGS AS NEEDED
    tokenizer_path = "data/tokenizer.json"
    checkpoint_path = "checkpoints/model_epoch1.pt"  # change to the checkpoint you want to evaluate
    model_config = {
        "embed_dim": 300,
        "n_layers": 3,
        "n_heads": 6,
        "d_ff": 1024,
        "context_len": 64
    }

    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model_from_checkpoint(checkpoint_path, tokenizer, model_config)

    metrics = evaluate_many(model, tokenizer, checkpoint_info=checkpoint_path, num_samples=50, max_new_tokens=50, temperature=1.0, top_k=40, sample=True, out_csv="generation_eval.csv")
    print("Evaluation finished.")
    print("Average perplexity (per token):", metrics["avg_perplexity"])
    print("Average BLEU:", metrics["avg_bleu"])
    print("Total generated tokens scored:", metrics["total_generated_tokens"])
