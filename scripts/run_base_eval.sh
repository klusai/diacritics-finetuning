#!/bin/bash
set -e
cd ~/codespace/phd/diacritics-finetuning-code
source .venv/bin/activate
export PYTHONPATH=.

echo "=== Waiting for LoRA queue to finish ==="
while ps aux | grep -v grep | grep "mlx_lm.*lora" > /dev/null 2>&1; do
    echo "$(date): LoRA training still running, waiting 60s..."
    sleep 60
done
echo "LoRA queue finished at $(date)"

echo "=== LLMic_v2 Base Evaluation (Single-line prompt) ==="
python scripts/eval_mlx.py \
  --model faur-ai/LLMic_v2 \
  --output-dir artifacts/base-llmic-v2-singleline \
  --name llmic-v2-base-singleline \
  --prompt-style single-line \
  --batch-size 32

echo "=== LLMic_v2 Base Evaluation (Few-shot prompt) ==="
python scripts/eval_mlx.py \
  --model faur-ai/LLMic_v2 \
  --output-dir artifacts/base-llmic-v2-fewshot \
  --name llmic-v2-base-fewshot \
  --prompt-style fewshot \
  --batch-size 32

echo "=== BASE EVAL COMPLETE at $(date) ==="
echo "Results:"
echo "--- Single-line ---"
python3 -c "
import json
r = json.load(open('artifacts/base-llmic-v2-singleline/results.json'))
for k, v in sorted(r.items()):
    if k == 'speed': continue
    print(f'  {k}: RA_CS_WL={v[\"RA_CS_WL\"][\"mean\"]:.4f}, DER={v[\"DER\"][\"mean\"]:.4f}')
"
echo "--- Few-shot ---"
python3 -c "
import json
r = json.load(open('artifacts/base-llmic-v2-fewshot/results.json'))
for k, v in sorted(r.items()):
    if k == 'speed': continue
    print(f'  {k}: RA_CS_WL={v[\"RA_CS_WL\"][\"mean\"]:.4f}, DER={v[\"DER\"][\"mean\"]:.4f}')
"
