import csv
import gc
from vllm import LLM, SamplingParams
import pandas as pd
import torch

models = [
    {"model": "/shared/storage-01/users/ishikaa2/sft_translation_models/llama_chinese/checkpoint-609", "shorthand": "llama_chinese_sft"},
    {"model": "/shared/storage-01/users/ishikaa2/sft_translation_models/llama_hindi/checkpoint-375", "shorthand": "llama_hindi_sft"},
    {"model": "/shared/storage-01/users/ishikaa2/sft_translation_models/qwen_chinese/checkpoint-609", "shorthand": "qwen_chinese_sft"},
    {"model": "/shared/storage-01/users/ishikaa2/sft_translation_models/qwen_hindi/checkpoint-375", "shorthand": "qwen_hindi_sft"},
    {"model": "CohereLabs/c4ai-command-r-08-2024", "shorthand": "command_r_32b"},
    {"model": "CohereLabs/c4ai-command-r7b-12-2024", "shorthand": "command_r_7b"},
    {"model": "meta-llama/Llama-3.2-1B", "shorthand": "llama_base"},
    {"model": "Qwen/Qwen2.5-3B", "shorthand": "qwen_base"},
]

if __name__ == "__main__":
    datasets = [
        {"df": 'Dataset/hindi-english_idioms.csv', "language": "Hindi"},
        {"df": 'Dataset/petci_chinese_english_improved.csv', "language": "Chinese"}
    ]

    sampling_params = SamplingParams(
        temperature=0.3,
        max_tokens=512
    )

    for model in models:
        llm = LLM(model['model'], tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.7, trust_remote_code=True)

        for dataset in datasets:
            lang = dataset['language']
            prompt = lambda idiom: f"Given the following idiom in {lang}, translate it semantically into English.\nIdiom: {idiom}\n### Semantic Translation:"

            df = pd.read_csv(dataset['df'])

            inputs = df['src'].apply(lambda x: prompt(x))
            outputs = llm.generate(inputs, sampling_params=sampling_params)
            outputs = [o.outputs[0].text.strip().split("\n")[0] for o in outputs]
        
            df['predicted'] = outputs
            df.to_csv(f"{model['shorthand']}-{lang}-outputs.csv", sep="|")
        
        del llm
            