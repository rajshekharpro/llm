# Decoder-Only Transformer Assignment (LLM2025)

## Overview
This project implements a decoder-only transformer language model trained on the TinyStories dataset (train split). It uses word-level tokenization and initializes embeddings from FastText vectors.

## Structure
- tokenizers/tokenizer.py: word-level tokenizer (builds vocab).
- utils/data_utils.py: data download/preprocessing, building embedding matrix from FastText.
- models/transformer.py: model implementation (LayerNorm, MHA, FFN, decoder blocks).
- training/train.py: training loop, saving checkpoints and plot.
- generation/generate.py & generation/beam.py: sampling and beam search.
- utils/plotting.py: plotting utilities.

## Requirements
Install dependencies:
