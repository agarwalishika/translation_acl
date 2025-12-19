from vllm import LLM, SamplingParams
import pandas as pd
import torch
import re

if __name__ == "__main__":
    models = [
        {"model": "meta-llama/Llama-3.2-1B", "shorthand": "llama-1b"},
        {"model": "Qwen/Qwen2.5-3B", "shorthand": "qwen-3b"},
        
    ]
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
            df = pd.read_csv(dataset['df'])

            # TODO: finish the methodology

            # TODO: parse the final output and store the final model predictions in a list of strings
            predicted_outputs = [...]

            # save the predicted output to a csv
            df['predicted'] = predicted_outputs
            df.to_csv(f"outputs/{model['shorthand']}-LIA-{lang}-outputs.csv")
        
        del llm

