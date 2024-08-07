import subprocess
import os
from pathlib import Path
import argparse
import lm_eval
from lm_eval.utils import handle_non_serializable
import json


tasks = [
    'professional_psychology',
    'management',
    'moral_scenarios',
    'security_studies',
    'sociology',
    'world_religions',
    'business_ethics',
    'elementary_mathematics',
    'high_school_computer_science',
    'global_facts',
    'high_school_mathematics',
    'high_school_world_history',
    "abstract_algebra",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_physics",
    "computer_security",
    "formal_logic",
    "logical_fallacies",
    "machine_learning",
    "moral_disputes",
    "moral_scenarios",
    "philosophy"
]

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, default=None, help="Name of the model to run")
args = parser.parse_args()
#TRANSFORMERS_OFFLINE = 0
#token='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'

complete_tasks = []
for t in tasks:
    complete_tasks.append(f"mmlu_{t}")

model = args.model


if model != 'tiiuae/falcon-rw-1b':
    model_trust = model+",trust_remote_code=True"
else:
    model_trust = model

results = lm_eval.simple_evaluate(
    model="hf",
    model_args="pretrained="+model_trust,
    tasks=complete_tasks,
    log_samples=False,
    batch_size=3
    #log_samples=True
)
results['results']['model_name'] = model


if not os.path.isdir(f'eval_scripts/output'):
    os.makedirs(f'eval_scripts/output')

with open(f"eval_scripts/output/mmlusubtasks_{model.split('/')[1]}.json", "w") as fp:
    json.dump(results, fp, indent=2, default=handle_non_serializable, ensure_ascii=False)
    print(f"Results saved to eval_scripts/output/mmlusubtasks_{model.split('/')[1]}.json")                                                                  

#dumped = json.dump(results , indent=2, default=handle_non_serializable, ensure_ascii=False)
#print(model, dumped)
#with open(f"output/mmlu2_{model}.json", "w") as fp:

