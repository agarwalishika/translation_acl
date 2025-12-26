import os
from vllm import LLM, SamplingParams
import pandas as pd
import torch

models = [
    ## grpo models
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Chinese_llama1b-da", "shorthand": "grpo_Chinese_llama1b-da"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Chinese_llama1b-qe-cons", "shorthand": "grpo_Chinese_llama1b-qe-cons"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Chinese_llama1b-qe-pos", "shorthand": "grpo_Chinese_llama1b-qe-pos"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Chinese_llama1b-qe-neg", "shorthand": "grpo_Chinese_llama1b-qe-neg"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Chinese_qwen3b-da", "shorthand": "grpo_Chinese_qwen3b-da"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Chinese_qwen3b-qe-cons", "shorthand": "grpo_Chinese_qwen3b-qe-cons"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Chinese_qwen3b-qe-pos", "shorthand": "grpo_Chinese_qwen3b-qe-pos"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Chinese_qwen3b-qe-neg", "shorthand": "grpo_Chinese_qwen3b-qe-neg"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Hindi_llama1b-da", "shorthand": "grpo_Hindi_llama1b-da"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Hindi_llama1b-qe-cons", "shorthand": "grpo_Hindi_llama1b-qe-cons"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Hindi_llama1b-qe-pos", "shorthand": "grpo_Hindi_llama1b-qe-pos"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Hindi_llama1b-qe-neg", "shorthand": "grpo_Hindi_llama1b-qe-neg"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Hindi_qwen3b-da", "shorthand": "grpo_Hindi_qwen3b-da"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Hindi_qwen3b-qe-cons", "shorthand": "grpo_Hindi_qwen3b-qe-cons"},
    {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_trains/Hindi_qwen3b-qe-pos", "shorthand": "grpo_Hindi_qwen3b-qe-pos"},
    # {"model": "/shared/storage-01/users/ishikaa2/grpo_translation_models/Hindi_qwen3b-qe-neg", "shorthand": "grpo_Hindi_qwen3b-qe-neg"},
    
    ## translation models
    # {"model": "CohereLabs/c4ai-command-r-08-2024", "shorthand": "command_r_32b"},
    # {"model": "CohereLabs/c4ai-command-r7b-12-2024", "shorthand": "command_r_7b"},
    
    ## base models
    # {"model": "meta-llama/Llama-3.1-8B", "shorthand": "llama_base"},
    # {"model": "Qwen/Qwen2.5-3B", "shorthand": "qwen_base"},

    # ## sft models
    # {"model": "/shared/storage-01/users/ishikaa2/sft_translation_models/qwen_chinese/checkpoint-375", "shorthand": "qwen_chinese_sft"},
    # {"model": "/shared/storage-01/users/ishikaa2/sft_translation_models/qwen_hindi/checkpoint-300", "shorthand": "qwen_hindi_sft"},
    # {"model": "/shared/storage-01/users/ishikaa2/sft_translation_models/llama_chinese/checkpoint-375", "shorthand": "llama_chinese_sft"},
    # {"model": "/shared/storage-01/users/ishikaa2/sft_translation_models/llama_hindi/checkpoint-300", "shorthand": "llama_hindi_sft"},
    
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
        llm = LLM(model['model'], tensor_parallel_size=torch.cuda.device_count(), gpu_memory_utilization=0.6, trust_remote_code=True)
        for dataset in datasets:
            for i in range(32):
                lang = dataset['language']
                filename = f"outputs/{model['shorthand']}-{lang}-outputs_{i}.csv"
                if os.path.exists(filename): continue

                print(f"\n\n\nHELLO I AM EVALUATING {filename}")

                prompt = lambda idiom: f"Given the following idiom in {lang}, translate it semantically into English.\nIdiom: {idiom}\nOutput your answer below:\n### Semantic Translation: "

                df = pd.read_csv(dataset['df'])
                if "hindi" in dataset['df']: df = df[800:]
                elif "chinese" in dataset['df']: df = df[1000:]
                else: 0/0

                inputs = df['src'].apply(lambda x: prompt(x))
                outputs = llm.generate(inputs, sampling_params=sampling_params)
                outputs = [o.outputs[0].text.strip().split("\n")[0] for o in outputs]
            
                df['predicted'] = outputs
                df.to_csv(filename, sep="|")
            
        del llm
            