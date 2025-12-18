import csv
import gc
from vllm import LLM, SamplingParams
import pandas as pd
import torch
from transformers import pipeline
from tqdm import tqdm

if __name__ == "__main__":
    datasets = [
        {"df": 'Dataset/hindi-english_idioms.csv', "language": "Hindi", "lang_code": "hin_Deva"},
        {"df": 'Dataset/petci_chinese_english_improved.csv', "language": "Chinese", "lang_code": "zho_Hant"}
    ]

    models = [
        "facebook/nllb-200-distilled-1.3B",
        "facebook/nllb-200-3.3B"
    ]

    for model in models:
        for dataset in datasets:
            df = pd.read_csv(dataset['df'])
            inf_pipeline = pipeline(task="translation", model=model, src_lang=dataset['lang_code'], tgt_lang="eng_Latn", dtype=torch.float16, device=0)

            inputs = list(df['src'])
            outputs = []
            bs = 8
            for i in tqdm(range(0, len(inputs), bs)):
                temp = inf_pipeline(inputs[i:i+bs])
                outputs.extend([o['translation_text'] for o in temp])
        
            lang = dataset['language']
            df['predicted'] = outputs
            df.to_csv(f"{model.split('/')[-1]}-{lang}-outputs.csv", sep="|")
        
            