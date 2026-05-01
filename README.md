# Diacritics Fine-Tuning

Standalone fine-tuning and evaluation pipeline for Romanian diacritic restoration using small open-weight language models.

## Overview

This project implements a comparative study of diacritic restoration approaches for Romanian text:

- **Data pipeline**: corpus loading from dexonline, diacritic stripping, configurable noise injection (typos, casing, cedilla mixing), train/val/test splitting with deduplication
- **Three model families**: lightweight supervised (dictionary, BiLSTM, CharCNN+BERT), sequence-to-sequence (ByT5), and decoder-only LLMs (LoRA fine-tuning)
- **Evaluation suite**: 10 metrics (RA/RER at character/word level, DER, hallucination rate), per-character F-scores, bootstrap confidence intervals, McNemar's test, inference speed benchmarking, and error analysis
- **Noise robustness**: evaluation at multiple noise levels to characterize when generative models outperform task-specific baselines

## Project Structure

```
diacritics-finetuning/
├── diacritics/                  # Main package
│   ├── data/                    # Corpus loading, stripping, noise injection
│   ├── models/                  # Model implementations (dictionary, BiLSTM, ...)
│   ├── training/                # Training configs and pipelines
│   └── evaluation/              # Metrics, per-char analysis, error analysis, speed
├── scripts/                     # CLI entry points
├── conf/                        # YAML configs (models, training hyperparameters)
├── data/                        # Generated data splits (gitignored)
└── artifacts/                   # Training outputs and results (gitignored)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Configuration

Set the `DEXONLINE_DATA_DIR` environment variable to point to your dexonline corpus:

```bash
export DEXONLINE_DATA_DIR=/path/to/dexonline
```

The directory should contain `crawler/json/` and `dlrlc/json/` subdirectories with the dexonline JSON files.

## Usage

### Prepare data

```bash
PYTHONPATH=. python scripts/prepare_data.py --output-dir data/splits
```

Generates train/val/test splits with deduplication and noisy test variants.

### Train models

```bash
# Dictionary baseline (most-frequent-form lookup)
PYTHONPATH=. python scripts/train.py dictionary --data-dir data/splits

# Character BiLSTM
PYTHONPATH=. python scripts/train.py bilstm --data-dir data/splits --epochs 5
```

### Evaluate

```bash
PYTHONPATH=. python scripts/evaluate.py \
    artifacts/model/predictions_test_clean.jsonl \
    data/splits/test_crawler_1000_clean.jsonl \
    --per-char --error-report
```

## Related

- **Paper**: [mihainadas/diacritics-finetuning-paper](https://github.com/mihainadas/diacritics-finetuning-paper) (private)
- **Prior work (prompting study)**: [Evaluating LLMs for Diacritic Restoration in Romanian Texts](https://arxiv.org/abs/2511.13182) (InnoComp 2025, Springer LNCS)
- **Summa**: [mihainadas/summa](https://github.com/mihainadas/summa) -- original evaluation codebase

## License

MIT
