# generation/generate.py
import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.transformer import DecoderOnlyTransformer

def top_k_logits(logits, k):
    if k is None:
        return logits
    v, ix = torch.topk(logits, k)
    out = logits.clone()
    out[out < v[:, [-1]]] = -1e10
    return out

def generate_sample(model, tokenizer, prompt_text, max_new_tokens=50, temperature=1.0, top_k=None, sample=True, device="cuda"):
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt_text)], dtype=torch.long).to(device)
    out_ids = model.generate(ids, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, sample=sample, tokenizer=tokenizer, device=device)
    return tokenizer.decode(out_ids[0].tolist())

# wrapper for batch evaluation
def evaluate_generation(model, tokenizer, val_texts, n_samples=50, device="cuda"):
    import evaluate as eval_lib
    bleu = eval_lib.load("bleu")
    total_ppl = 0.0
    samples = val_texts[:n_samples]
    results = []
    for t in tqdm(samples):
        toks = t.split()
        prompt = " ".join(toks[:5]) if len(toks) > 5 else t
        gen = generate_sample(model, tokenizer, prompt, device=device)
        # compute perplexity per token: not exact, but we can compute model negative log-likelihood on generated continuation
        # Here we'll approximate by re-scoring generated continuation conditioned on prompt
        # TODO: implement exact per-token perplexity calc
        results.append((prompt, t, gen))
    # compute BLEU
    refs = [[r.split()] for _, r, _ in results]
    hyps = [g.split() for _, _, g in results]
    bleu_res = bleu.compute(predictions=hyps, references=refs)
    return results, bleu_res
