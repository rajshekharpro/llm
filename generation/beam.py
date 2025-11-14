# generation/beam.py

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
import math
from tqdm import tqdm

def beam_search(model, input_ids, k=5, max_new_tokens=50, tokenizer=None, device="cuda"):
    # input_ids: (B, T) - we'll run beam per example (batch size 1 for simplicity)
    model.eval()
    B = input_ids.size(0)
    assert B == 1, "beam_search currently supports batch size 1 for simplicity"

    device = input_ids.device
    input_ids = input_ids.to(device)
    # Each beam: tuple (score (logprob), token_ids tensor)
    beams = [(0.0, input_ids)]
    completed = []

    for step in range(max_new_tokens):
        new_beams = []
        for score, seq in beams:
            logits, _, _ = model(seq)
            logits = logits[:, -1, :]  # (1, V)
            log_probs = F.log_softmax(logits, dim=-1).squeeze(0)  # (V,)
            topk_logp, topk_idx = torch.topk(log_probs, k)
            for i in range(k):
                new_score = score + topk_logp[i].item()
                new_seq = torch.cat([seq, topk_idx[i].unsqueeze(0).unsqueeze(0)], dim=1)
                new_beams.append((new_score, new_seq))
        # keep top k beams
        new_beams = sorted(new_beams, key=lambda x: x[0], reverse=True)[:k]
        beams = []
        for sc, s in new_beams:
            # stop if last token is EOS
            if tokenizer and s[0, -1].item() == tokenizer.word2idx[tokenizer.eos_token]:
                completed.append((sc, s))
            else:
                beams.append((sc, s))
        if len(beams) == 0:
            break
    if len(completed) == 0:
        completed = beams
    completed = sorted(completed, key=lambda x: x[0], reverse=True)
    best = completed[0][1]
    return best
