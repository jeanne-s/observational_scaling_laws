import subprocess
import os
from pathlib import Path
import argparse
import lm_eval
from lm_eval.utils import handle_non_serializable
import json


tasks = [
    "advanced_ai_risk",
    "persona",
    "sycophancy",
    "winogenerated"
]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, default=None, help="Name of the model to run")
args = parser.parse_args()

model = args.model


if model != 'tiiuae/falcon-rw-1b':
    model_trust = model+",trust_remote_code=True"
else:
    model_trust = model

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained="+model_trust,
    tasks=tasks,
    log_samples=False,
    batch_size=3
)
results['results']['model_name'] = model


if not os.path.isdir(f'eval_scripts/output'):
    os.makedirs(f'eval_scripts/output')

with open(f"eval_scripts/model_written_evals/{model.split('/')[1]}.json", "w") as fp:
    json.dump(results, fp, indent=2, default=handle_non_serializable, ensure_ascii=False)
    print(f"Results saved to eval_scripts/model_written_evals/{model.split('/')[1]}.json")                                                                  