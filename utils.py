import pandas as pd


BENCHMARKS = ['MMLU', 'ARC-C', 'HellaSwag', 'Winograd', 'TruthfulQA', 'GSM8K', 'XWinograd', 'HumanEval']
MODEL_FAMILIES = {'pythia': 'Pythia',
                  'Llama-2': 'Llama-2',
                  'llama-': 'Llama',
                  'Qwen1.5': 'Qwen-1.5',
                  'Qwen-': 'Qwen',
                  'falcon': 'Falcon',
                  'bloom': 'BLOOM',
                  'gpt-': 'GPT-Neo/J',
                  'opt': 'OPT',
                  'xglm': 'XGLM',
                  'CodeLlama': 'CodeLlama',
                  'starcoderbase': 'StarCoder'
}
ADDITIONAL_FAMILIES = {'starcoder2': 'StarCoder2',
                       'deepseek-coder': 'DeepSeek-Coder',
                       'open_llama': 'OpenLlama',
                       'stablelm': 'StableLM',
                       'rwkv': 'RWKV'
}

def load_leaderboard_data(family_name: str = None):
    leaderboard = pd.read_csv('leaderboards/base_llm_benchmark_eval.csv')
    if family_name is not None:
        leaderboard = leaderboard[leaderboard['Model'].str.contains(family_name)]
    return leaderboard