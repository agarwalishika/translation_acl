import csv
import gc
from vllm import LLM, SamplingParams
import pandas as pd

# 1️⃣ Define models
models = [
    {"name": "facebook/nllb-200-distilled-1.3B", "type": "seq2seq"},
    {"name": "CohereLabs/c4ai-command-r-08-2024", "type": "causal"},
    {"name": "CohereLabs/c4ai-command-r7b-12-2024", "type": "causal"}
]

# 2️⃣ Define idioms per language
hindi_idioms = pd.read_csv('Dataset/hindi-english_idioms.csv')['src']
chinese_idioms = pd.read_csv('Dataset/petci_chinese_english_improved.csv')['srv']
idioms_per_language = {
    "Hindi": hindi_idioms,
    "Chinese": chinese_idioms
}

# NLLB language codes
lang_map = {
    "Hindi": "hin_Deva",
    "Chinese": "cmn_Hans"
}

# 3️⃣ Generation parameters
sampling_params = SamplingParams(
    temperature=0.3,
    max_output_tokens=96
)

# 4️⃣ Translation function
def translate_vllm(llm, prompt, lang_code=None):
    if lang_code:
        output = llm.generate([prompt], sampling_params=sampling_params,
                              forced_bos_token_id=llm.tokenizer.lang_code_to_id[lang_code])
    else:
        output = llm.generate([prompt], sampling_params=sampling_params)
    return output[0].outputs[0].text.strip()

# 5️⃣ Main loop
for model_info in models:
    model_name = model_info["name"]
    model_type = model_info["type"]
    
    print(f"Loading model {model_name}...")
    llm = LLM(model=model_name)  # load model
    
    for language, idioms in idioms_per_language.items():
        results = []
        for idiom in idioms:
            prompt = f"Given the following idiom in {language}, translate it semantically into English: {idiom}"
            
            # Only NLLB needs the language code of the source
            lang_code = lang_map[language] if "nllb" in model_name.lower() else None
            translation = translate_vllm(llm, prompt, lang_code)
            
            results.append({"idiom": idiom, "translation": translation})
        
        # Save CSV
        safe_model_name = model_name.replace("/", "_")
        csv_file = f"{safe_model_name}_{language}_to_English.csv"
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["idiom", "translation"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Saved {csv_file}")
    
    # Unload model to free memory
    del llm
    gc.collect()
    print(f"Unloaded model {model_name}\n")

print("All done!")
