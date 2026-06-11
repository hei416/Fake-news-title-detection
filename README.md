# Fake News Title Detection

A BERT-based fake news classification system built for the Introduction to AI course.
Classifies news headlines as real or fake using the FakeNewsNet dataset,
combining deep learning with heuristic rule-based detection for better interpretability.

## Features
- Fine-tuned BERT model on PolitiFact + GossipCop subsets
- Full ML pipeline: preprocessing, training, evaluation, threshold optimization
- Interactive analyzer combining model predictions with heuristic rules
- Evaluation metrics: accuracy, precision, recall, F1

## Requirements
Python 3.8+, PyTorch 2.0+, Transformers, pandas, numpy, sklearn, matplotlib, seaborn

## Dataset
FakeNewsNet dataset (PolitiFact + GossipCop subsets)
Place CSV files in ./dataset/ directory
Expected format: columns for text and label (0=real, 1=fake)

## Quick Start
Run all cells in order in the notebook.
Training takes ~30-45 min on GPU (CUDA recommended).
Model saves to multimodal_fake_news_detector.pt

## Key Parameters
- Batch size: 8 (with gradient accumulation × 2)
- Max sequence length: 512 tokens
- Epochs: 15 (with early stopping)
