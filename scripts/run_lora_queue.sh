#!/bin/bash
set -e
cd ~/codespace/phd/diacritics-finetuning-code
source .venv/bin/activate
export PYTHONPATH=.
git pull --quiet

echo "=== Waiting for current Qwen3 training to finish ==="
while ps aux | grep -v grep | grep "mlx_lm.*lora" > /dev/null 2>&1; do sleep 30; done
echo "Current training done."

echo "=== Regenerating training data with clean prompt format ==="
python3 -c "
from diacritics.models.decoder_lm import prepare_mlx_dataset
from pathlib import Path
import json

train = [json.loads(l) for l in open('data/splits/train.jsonl')]
pairs = [(r['input'], r['target']) for r in train]
print(f'Preparing MLX dataset: {len(pairs)} pairs')
prepare_mlx_dataset(pairs, Path('artifacts/lora/qwen3-1.7b-v2/data'), val_ratio=0.01)
"

echo "=== Training Qwen3-1.7B v2 (clean prompt) ==="
cat > artifacts/lora/qwen3-1.7b-v2/config.yaml << EOF
model: "Qwen/Qwen3-1.7B"
data: "artifacts/lora/qwen3-1.7b-v2/data"
adapter_path: "artifacts/lora/qwen3-1.7b-v2/adapters"
fine_tune_type: "lora"
train: true
mask_prompt: true
grad_checkpoint: true
batch_size: 4
iters: 5000
learning_rate: 0.0002
steps_per_eval: 500
steps_per_report: 50
save_every: 1000
max_seq_length: 512
num_layers: -1
lora_parameters:
  rank: 16
  alpha: 32.0
  dropout: 0.0
  scale: 2.0
EOF

python -m mlx_lm lora -c artifacts/lora/qwen3-1.7b-v2/config.yaml 2>&1 | tee artifacts/lora/qwen3-1.7b-v2/train.log

echo "=== Evaluating Qwen3-1.7B v2 ==="
python3 -c "
import logging, json
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

from diacritics.models.decoder_lm import MLXDecoderLM
from scripts.train import evaluate_model

model = MLXDecoderLM('Qwen/Qwen3-1.7B', adapter_path='artifacts/lora/qwen3-1.7b-v2/adapters')
out = Path('artifacts/lora/qwen3-1.7b-v2')
evaluate_model(model.predict, Path('data/splits'), out, 'qwen3-1.7b-v2')
"

echo "=== Qwen3 v2 complete. Starting LLMic_v2 ==="

echo "=== Regenerating data for LLMic_v2 ==="
python3 -c "
from diacritics.models.decoder_lm import prepare_mlx_dataset
from pathlib import Path
import json

train = [json.loads(l) for l in open('data/splits/train.jsonl')]
pairs = [(r['input'], r['target']) for r in train]
prepare_mlx_dataset(pairs, Path('artifacts/lora/llmic-v2/data'), val_ratio=0.01)
"

echo "=== Training LLMic_v2 ==="
cat > artifacts/lora/llmic-v2/config.yaml << EOF
model: "faur-ai/LLMic_v2"
data: "artifacts/lora/llmic-v2/data"
adapter_path: "artifacts/lora/llmic-v2/adapters"
fine_tune_type: "lora"
train: true
mask_prompt: true
grad_checkpoint: true
batch_size: 2
iters: 5000
learning_rate: 0.0002
steps_per_eval: 500
steps_per_report: 50
save_every: 1000
max_seq_length: 512
num_layers: -1
lora_parameters:
  rank: 16
  alpha: 32.0
  dropout: 0.0
  scale: 2.0
EOF

python -m mlx_lm lora -c artifacts/lora/llmic-v2/config.yaml 2>&1 | tee artifacts/lora/llmic-v2/train.log

echo "=== Evaluating LLMic_v2 ==="
python3 -c "
import logging, json
from pathlib import Path
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')

from diacritics.models.decoder_lm import MLXDecoderLM
from scripts.train import evaluate_model

model = MLXDecoderLM('faur-ai/LLMic_v2', adapter_path='artifacts/lora/llmic-v2/adapters')
out = Path('artifacts/lora/llmic-v2')
evaluate_model(model.predict, Path('data/splits'), out, 'llmic-v2')
"

echo "=== ALL LORA TRAINING COMPLETE ==="
