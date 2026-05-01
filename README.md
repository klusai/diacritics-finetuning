# Diacritics Fine-Tuning

Standalone fine-tuning and evaluation pipeline for Romanian diacritic restoration using small open-weight language models.

## Overview

This project implements:
- Data preparation from dexonline corpus (diacritic stripping, noise injection, train/val/test splitting)
- Three model families: lightweight supervised (BiLSTM, CharCNN+BERT), sequence-to-sequence (ByT5), and decoder-only LLMs (LoRA/full fine-tuning)
- Comprehensive evaluation suite covering word/character accuracy, DER, hallucination rate, and per-character F-scores
- Noise robustness experiments and error analysis

## Architecture Roadmap

| Directory | Planned Modules | Purpose |
|-----------|----------------|---------|
| `diacritics/data/` | `dexonline.py`, `strip.py`, `noise.py` | Corpus loading/preprocessing, diacritic stripping for training pairs, noise injection (typos, casing) for robustness experiments |
| `diacritics/models/` | `bilstm.py`, `bert_classifier.py`, `byt5.py`, `decoder_lm.py` | Model wrappers: character BiLSTM baseline, CharCNN+BERT token classification, ByT5 seq2seq, decoder-only LLM (LoRA/full) |
| `diacritics/training/` | `lora.py`, `full.py`, `config.py` | LoRA fine-tuning (PEFT), full fine-tuning, training hyperparameter configs |
| `diacritics/evaluation/` | `metrics.py`, `per_char.py`, `error_analysis.py` | WA/CA/DER/hallucination metrics, per-character F-scores (ă â î ș ț), error categorization (â/î confusion, Ș/Ț) |
| `scripts/` | `prepare_data.py`, `train.py`, `evaluate.py`, `error_report.py` | CLI entry points |
| `conf/` | `models.yaml`, `training.yaml` | Model definitions, default training hyperparameters |

## Installation

```bash
pip install -r requirements.txt
```

## Related Repositories

- **Paper:** [diacritics-finetuning-paper](../diacritics-finetuning-paper)
- **Prior work (prompting):** [llm-diacritics-paper](../llm-diacritics-paper)
- **Summa (evaluation code):** [github.com/mihainadas/summa](https://github.com/mihainadas/summa)
- **dexonline:** [dexonline.ro](https://dexonline.ro)

## License

MIT
