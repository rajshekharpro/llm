# generation/run_generate.py
import torch
import json
from models.transformer import DecoderOnlyTransformer
from word_tokenizer.tokenizer import WordTokenizer   # adjust if your folder name differs
from generation.generate import generate_sample
import os

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_tokenizer(json_path="data/tokenizer.json"):
    tok = WordTokenizer()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    tok.word2idx = data["word2idx"]
    tok.idx2word = {int(i):w for i,w in enumerate(tok.word2idx.keys())} \
        if isinstance(list(tok.word2idx.keys())[0], int) else {v:k for k,v in tok.word2idx.items()}
    return tok

def load_model(checkpoint_path, tokenizer, config):
    # config should match what you used in training
    model = DecoderOnlyTransformer(
        vocab_size=tokenizer.vocab_size,
        embed_dim=config["embed_dim"],
        n_layers=config["n_layers"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        context_len=config["context_len"],
        embedding_weights=None,   # set if you have saved embedding weights; otherwise None
        freeze_embeddings=False
    ).to(DEVICE)

    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    # if you saved state dict under 'model_state'
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    return model

if __name__ == "__main__":
    # Edit these paths as needed
    tokenizer_path = "data/tokenizer.json"
    checkpoint = "checkpoints/model_epoch1.pt"   # change to the checkpoint you want
    config = {
        "embed_dim": 300,
        "n_layers": 3,
        "n_heads": 6,
        "d_ff": 1024,
        "context_len": 64
    }

    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(checkpoint, tokenizer, config)

    # example generation
    prompt = "Once upon a"
    generated = generate_sample(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=40, sample=True, device=DEVICE)
    print("PROMPT:", prompt)
    print("GENERATED:", generated)
