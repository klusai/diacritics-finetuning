#!/bin/bash
set -e
cd ~/codespace/phd/diacritics-finetuning-code
source .venv/bin/activate
export PYTHONPATH=.

echo "=== Waiting for LoRA queue to finish ==="
while ps aux | grep -v grep | grep "lora_queue_v3" > /dev/null 2>&1; do sleep 60; done
echo "LoRA queue finished at $(date)"

echo "=== Evaluating Qwen3 1.7B prompting predictions ==="
python3 -c "
import json, logging
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger('qwen3_prompting_eval')

from diacritics.evaluation.metrics import evaluate_batch, aggregate_scores
from diacritics.evaluation.per_char import per_char_scores, format_per_char_report
from diacritics.evaluation.error_analysis import analyze_errors, ai_confusion_rates, format_error_report

prompting_dir = Path('artifacts/prompting')
data_dir = Path('data/splits')

for pred_file in sorted(prompting_dir.glob('predictions_qwen3*')):
    model_test = pred_file.stem.replace('predictions_', '')
    if 'crawler' in model_test:
        gold_file = data_dir / 'test_crawler_1000_clean.jsonl'
    else:
        gold_file = data_dir / 'test_dlrlc_1000_clean.jsonl'

    preds = [json.loads(l) for l in open(pred_file) if l.strip()]
    golds = {r['id']: r['target'] for r in [json.loads(l) for l in open(gold_file)]}

    gold_list, pred_list = [], []
    for p in preds:
        if p['id'] in golds:
            gold_list.append(golds[p['id']])
            pred_list.append(p['prediction'])

    per_item = evaluate_batch(gold_list, pred_list)
    agg = aggregate_scores(per_item)
    logger.info('%s: RA_CS_WL=%.4f, DER=%.4f', model_test, agg['RA_CS_WL']['mean'], agg['DER']['mean'])

    # Update the prompting results.json
    results_path = prompting_dir / 'results.json'
    all_results = json.load(open(results_path)) if results_path.exists() else {}
    all_results[model_test] = agg
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    cs = per_char_scores(gold_list, pred_list)
    with open(prompting_dir / f'per_char_{model_test}.md', 'w') as f:
        f.write(f'# {model_test}\n\n')
        f.write(format_per_char_report(cs))

    es = analyze_errors(gold_list, pred_list)
    ai = ai_confusion_rates(gold_list, pred_list)
    with open(prompting_dir / f'error_{model_test}.md', 'w') as f:
        f.write(f'# {model_test}\n\n')
        f.write(format_error_report(es))
        if ai:
            f.write('\n\n### a/i Confusion\n')
            for k, v in sorted(ai.items()):
                f.write(f'- {k}: {v}\n')
"

echo "=== Checking LLMic_v2 license ==="
python3 -c "
from huggingface_hub import model_info
info = model_info('faur-ai/LLMic_v2')
print(f'LLMic_v2 license: {info.card_data.license if info.card_data else \"not found\"}')
print(f'Tags: {info.tags}')
" 2>/dev/null || echo "Could not fetch LLMic_v2 license info automatically"

echo "=== Exporting results to paper repo ==="
PAPER_RESULTS=~/codespace/phd/diacritics-finetuning-paper/data/results

# Create dirs
for model in dictionary bilstm bert_v2 byt5 prompting; do
    mkdir -p $PAPER_RESULTS/$model
done
mkdir -p $PAPER_RESULTS/lora/qwen3-1.7b-v2
mkdir -p $PAPER_RESULTS/lora/llmic-v2
mkdir -p $PAPER_RESULTS/base-llmic-v2

# Copy results, reports
for model in dictionary bilstm bert_v2 byt5; do
    cp artifacts/$model/results.json $PAPER_RESULTS/$model/
    cp artifacts/$model/per_char_report.md $PAPER_RESULTS/$model/ 2>/dev/null || true
    cp artifacts/$model/error_report.md $PAPER_RESULTS/$model/ 2>/dev/null || true
done

# Prompting
cp artifacts/prompting/results.json $PAPER_RESULTS/prompting/
cp artifacts/prompting/per_char_*.md $PAPER_RESULTS/prompting/ 2>/dev/null || true
cp artifacts/prompting/error_*.md $PAPER_RESULTS/prompting/ 2>/dev/null || true

# LoRA
for lora_model in qwen3-1.7b-v2 llmic-v2; do
    if [ -f artifacts/lora/$lora_model/results.json ]; then
        cp artifacts/lora/$lora_model/results.json $PAPER_RESULTS/lora/$lora_model/
        cp artifacts/lora/$lora_model/per_char_report.md $PAPER_RESULTS/lora/$lora_model/ 2>/dev/null || true
        cp artifacts/lora/$lora_model/error_report.md $PAPER_RESULTS/lora/$lora_model/ 2>/dev/null || true
        cp artifacts/lora/$lora_model/config.yaml $PAPER_RESULTS/lora/$lora_model/ 2>/dev/null || true
    fi
done

# Base LLMic_v2 (prompted ablation) -- copy whichever prompt variant performed better
for variant in base-llmic-v2-singleline base-llmic-v2-fewshot; do
    if [ -f artifacts/$variant/results.json ]; then
        cp artifacts/$variant/results.json $PAPER_RESULTS/base-llmic-v2/${variant}_results.json
        cp artifacts/$variant/per_char_report.md $PAPER_RESULTS/base-llmic-v2/${variant}_per_char_report.md 2>/dev/null || true
        cp artifacts/$variant/error_report.md $PAPER_RESULTS/base-llmic-v2/${variant}_error_report.md 2>/dev/null || true
        cp artifacts/$variant/examples.jsonl $PAPER_RESULTS/base-llmic-v2/${variant}_examples.jsonl 2>/dev/null || true
    fi
done

# Dataset stats
cp data/splits/stats.json $PAPER_RESULTS/dataset_stats.json 2>/dev/null || true

# Generate training summaries
python3 -c "
import json, re
from pathlib import Path

paper_results = Path('$PAPER_RESULTS')

summaries = {
    'dictionary': {'type': 'lookup', 'training_time_sec': 3, 'params': 'N/A (word lookup)', 'hardware': 'M3 Ultra'},
    'bilstm': {
        'type': 'character BiLSTM', 'params': '2,395,358', 'hardware': 'M3 Ultra MPS',
        'epochs': 5, 'batch_size': 256, 'lr': 0.001,
    },
    'bert_v2': {
        'type': 'BERT token classification (RoBERT-base)', 'params': '115,071,756', 'hardware': 'M3 Ultra MPS',
        'epochs': 3, 'batch_size': 16, 'lr': 3e-5,
    },
    'byt5': {
        'type': 'ByT5-small seq2seq', 'params': '300M', 'hardware': 'M3 Ultra MPS',
        'epochs': 3, 'batch_size': 8, 'lr': 1e-3, 'train_samples': 50000,
        'note': 'Trained on 50k subset (not full 300k) due to byte-level tokenization cost',
    },
}

# Extract training times from logs
log_patterns = {
    'bilstm': 'artifacts/bilstm_train.log',
    'bert_v2': 'artifacts/bert_v2_train.log',
    'byt5': 'artifacts/byt5_train.log',
}
for model, log_path in log_patterns.items():
    try:
        log = open(log_path).read()
        time_match = re.search(r'completed in ([\d.]+) seconds', log) or re.search(r'Done in ([\d.]+) min', log)
        if time_match:
            val = float(time_match.group(1))
            if 'min' in (time_match.group(0)):
                summaries[model]['training_time_min'] = round(val, 1)
            else:
                summaries[model]['training_time_min'] = round(val / 60, 1)
        
        losses = re.findall(r'Epoch (\d+)/\d+: loss=([\d.]+)', log)
        if losses:
            summaries[model]['epoch_losses'] = {int(e): float(l) for e, l in losses}
    except:
        pass

# LoRA summaries
for lora_name in ['qwen3-1.7b-v2', 'llmic-v2']:
    log_path = f'artifacts/lora/{lora_name}/train.log'
    try:
        log = open(log_path).read()
        val_losses = re.findall(r'Iter (\d+): Val loss ([\d.]+)', log)
        train_losses = re.findall(r'Iter (\d+): Train loss ([\d.]+)', log)
        summary = {
            'type': 'LoRA fine-tuned decoder',
            'hardware': 'M3 Ultra (MLX)',
            'lora_rank': 16, 'lora_alpha': 32,
            'iters': 5000, 'batch_size': 4 if '1.7b' in lora_name else 2,
        }
        if val_losses:
            summary['val_losses'] = {int(i): float(l) for i, l in val_losses}
        if train_losses:
            summary['final_train_loss'] = float(train_losses[-1][1])
        summaries[f'lora/{lora_name}'] = summary
    except:
        pass

# Base LLMic_v2 ablation (no training, just inference)
summaries['base-llmic-v2'] = {
    'type': 'Base decoder (prompted, no fine-tuning)',
    'model': 'faur-ai/LLMic_v2',
    'params': '3B',
    'hardware': 'M3 Ultra (MLX)',
    'training': 'none',
    'note': 'Ablation to isolate LoRA contribution from Romanian pretraining',
}

for model, summary in summaries.items():
    out_path = paper_results / model / 'training_summary.json'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {out_path}')
"

echo "=== Committing paper repo ==="
cd ~/codespace/phd/diacritics-finetuning-paper
git add data/results/
git commit -m "Add experimental results from all models (dictionary, BiLSTM, BERT, ByT5, prompting, LoRA, base ablation)" 2>/dev/null || echo "Nothing to commit"
git push 2>/dev/null || echo "Push failed (may need auth)"

echo "=== Committing code repo ==="
cd ~/codespace/phd/diacritics-finetuning-code
git add scripts/
git commit -m "Add post-training pipeline script" 2>/dev/null || echo "Nothing to commit"
git push 2>/dev/null || echo "Push failed"

echo "=== POST-TRAINING PIPELINE COMPLETE at $(date) ==="
