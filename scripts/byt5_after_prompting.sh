#!/bin/bash
set -e
echo "Waiting for prompting queue to finish..."
while ps aux | grep -v grep | grep "run_all_prompting.sh" > /dev/null; do sleep 60; done
echo "Prompting done. Waiting 60s for Ollama to unload models..."
sleep 60

cd ~/codespace/phd/diacritics-finetuning-code
source .venv/bin/activate
export PYTHONPATH=.
export PYTORCH_ENABLE_MPS_FALLBACK=1

python3 -c "
import json, time, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(message)s')
logger = logging.getLogger('byt5')
from diacritics.models.byt5 import ByT5Model
train = [json.loads(l) for l in open('data/splits/train_50k.jsonl')]
pairs = [(r['input'], r['target']) for r in train]
logger.info('ByT5 50k MPS (post-prompting): %d train', len(pairs))
model = ByT5Model(max_length=384)
start = time.time()
model.train(pairs, epochs=3, lr=1e-3, batch_size=8, use_cpu=False)
elapsed = time.time() - start
logger.info('Done in %.1f min', elapsed/60)
model.save('artifacts/byt5')
from pathlib import Path
from scripts.train import evaluate_model
evaluate_model(model.predict, Path('data/splits'), Path('artifacts/byt5'), 'byt5-small')
"
