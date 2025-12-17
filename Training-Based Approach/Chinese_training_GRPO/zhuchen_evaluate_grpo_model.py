#!/usr/bin/env python3
"""
GRPO Model Evaluation Script
Evaluates the trained GRPO model on the test set and compares with baseline models.
"""

import os
import json
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM
from comet import download_model, load_from_checkpoint
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Suppress Lightning logs
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTNING_FABRIC_DISABLE_TIPS"] = "1"
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)

# ===== Configuration =====
# Models to evaluate (add/remove as needed)
MODELS_TO_EVALUATE = {
    "qwen2.5_3b_grpo": {
        "path": "/projects/bfga/zshao7/qwen2.5-3b-grpo-comet-da-v3",
        "description": "Qwen2.5-3B GRPO (v3)"
    },
    "llama3.2_3b_grpo": {
        "path": "/projects/bfga/zshao7/llama3.2-3b-grpo-comet-da-v1",
        "description": "LLaMA 3.2-3B GRPO (v1)"
    }
}

# Use the first model's test set as reference (assumes same split)
PRIMARY_MODEL = "llama3.2_3b_grpo"  # Changed to use LLaMA's test set
TEST_CSV_PATH = f"{MODELS_TO_EVALUATE[PRIMARY_MODEL]['path']}/splits/test.csv"

BASELINE_FILES = {
    "qwen2.5_baseline": "/u/zshao7/homework/comet_results_3mt.csv",
    "llama3.2_3b_baseline": "/u/zshao7/homework/comet_results_3mt_llama3p2_3b.csv"
}
OUTPUT_DIR = "/u/zshao7/homework/grpo_evaluation_results"

# Generation parameters (same as training)
MAX_NEW_TOKENS = 96
GEN_TEMPERATURE = 0.8
GEN_TOP_P = 0.9
NUM_SAMPLES = 3  # Generate 3 samples per source for fair comparison with baseline

# Evaluation control
SKIP_EXISTING_RESULTS = True  # Set to False to re-evaluate models even if results exist

# Device setup
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

print("="*80)
print("GRPO Model Evaluation (Multi-Model)")
print("="*80)
print(f"Device: {DEVICE}")
print(f"Dtype: {DTYPE}")
print(f"Skip existing results: {SKIP_EXISTING_RESULTS}")
print(f"\nModels to evaluate:")
for model_name, model_info in MODELS_TO_EVALUATE.items():
    model_status = "✓" if os.path.exists(model_info['path']) else "⚠️ NOT FOUND"
    results_path = os.path.join(OUTPUT_DIR, f"{model_name}_test_results.csv")
    cache_status = " [CACHED]" if os.path.exists(results_path) and SKIP_EXISTING_RESULTS else ""
    print(f"  • {model_name}: {model_info['description']} [{model_status}]{cache_status}")
print(f"\nTest set: {TEST_CSV_PATH}")
print(f"Output directory: {OUTPUT_DIR}")
print("="*80)

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===== Step 1: Load Test Set =====
print("\n[1/6] Loading test set...")
test_df = pd.read_csv(TEST_CSV_PATH)
print(f"  → Test set size: {len(test_df)}")
print(f"  → Columns: {list(test_df.columns)}")

# ===== Step 2: Extract Baseline Scores for Test Set =====
print("\n[2/6] Extracting baseline scores for test set samples...")

baseline_results = {}
for baseline_name, baseline_path in BASELINE_FILES.items():
    if not os.path.exists(baseline_path):
        print(f"  ⚠️  Baseline file not found: {baseline_path}")
        continue

    print(f"  → Loading {baseline_name}...")
    baseline_df = pd.read_csv(baseline_path)

    # Match test samples with baseline by 'src' field
    matched = test_df.merge(baseline_df, on='src', how='left', suffixes=('', '_baseline'))

    # Count how many test samples were found in baseline
    matched_count = matched['DA_mt_best'].notna().sum()
    print(f"    • Matched {matched_count}/{len(test_df)} test samples")

    if matched_count > 0:
        # Extract DA scores
        da_scores = matched['DA_mt_best'].dropna().values

        # Extract QE scores if available
        if 'QE_mt_best' in matched.columns:
            qe_scores = matched['QE_mt_best'].dropna().values
            has_qe = len(qe_scores) > 0
        else:
            qe_scores = np.array([])
            has_qe = False

        baseline_results[baseline_name] = {
            'matched_count': matched_count,
            'avg_da': float(np.mean(da_scores)),
            'std_da': float(np.std(da_scores)),
            'min_da': float(np.min(da_scores)),
            'max_da': float(np.max(da_scores)),
            'median_da': float(np.median(da_scores)),
            'da_scores': da_scores.tolist(),
            'matched_df': matched[matched['DA_mt_best'].notna()].copy()
        }

        if has_qe:
            baseline_results[baseline_name].update({
                'avg_qe': float(np.mean(qe_scores)),
                'std_qe': float(np.std(qe_scores)),
                'min_qe': float(np.min(qe_scores)),
                'max_qe': float(np.max(qe_scores)),
                'median_qe': float(np.median(qe_scores)),
                'qe_scores': qe_scores.tolist()
            })

        print(f"    • Avg DA score: {baseline_results[baseline_name]['avg_da']:.4f} ± {baseline_results[baseline_name]['std_da']:.4f}")
        print(f"    • Range: [{baseline_results[baseline_name]['min_da']:.4f}, {baseline_results[baseline_name]['max_da']:.4f}]")
        if has_qe:
            print(f"    • Avg QE score: {baseline_results[baseline_name]['avg_qe']:.4f} ± {baseline_results[baseline_name]['std_qe']:.4f}")

# ===== Helper Function: Evaluate Single Model =====
def evaluate_single_model(model_name, model_path, test_df, da_model, qe_model, num_samples=NUM_SAMPLES, skip_existing=True):
    """Evaluate a single GRPO model on the test set"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    # Check if results already exist
    results_path = os.path.join(OUTPUT_DIR, f"{model_name}_test_results.csv")
    if skip_existing and os.path.exists(results_path):
        print(f"  ℹ️  Results already exist at {results_path}")
        print(f"  → Loading existing results (set skip_existing=False to re-evaluate)")
        try:
            results_df = pd.read_csv(results_path)
            print(f"  → Loaded {len(results_df)} results from cache")
            print(f"  → Avg DA (best): {results_df['DA_mt_best'].mean():.4f} ± {results_df['DA_mt_best'].std():.4f}")
            if 'QE_mt_best' in results_df.columns:
                print(f"  → Avg QE (best): {results_df['QE_mt_best'].mean():.4f} ± {results_df['QE_mt_best'].std():.4f}")
            return results_df
        except Exception as e:
            print(f"  ⚠️  Failed to load existing results: {e}")
            print(f"  → Re-evaluating model...")

    # Check if model exists
    if not os.path.exists(model_path):
        print(f"  ⚠️  Model not found at {model_path}")
        print(f"  Skipping this model...")
        return None

    # Load model
    print(f"  → Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=DTYPE,
        trust_remote_code=True
    )
    model.eval()
    print(f"  → Model loaded successfully")
    print(f"  → Model device: {next(model.parameters()).device}")

    # Generate translations
    print(f"  → Generating translations for {len(test_df)} test samples...")
    results_list = []

    for idx, row in tqdm(test_df.iterrows(), total=len(test_df), desc=f"  {model_name}"):
        src = row['src']
        ref = row['true_meaning']
        literal = row.get('literal_translation', '')

        # Format prompt
        prompt = format_prompt(src, tokenizer)

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        input_len = inputs.input_ids.shape[1]

        # Generate multiple samples
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=GEN_TEMPERATURE,
                top_p=GEN_TOP_P,
                num_return_sequences=num_samples,
                pad_token_id=tokenizer.eos_token_id
            )

        # Decode translations
        translations = []
        for i in range(num_samples):
            continuation = outputs[i][input_len:]
            text = tokenizer.decode(continuation, skip_special_tokens=True).strip()
            # Remove "English:" prefix if present
            if text.lower().startswith("english:"):
                text = text[len("english:"):].strip()
            translations.append(text)

        # Calculate COMET-DA scores (with reference)
        comet_da_items = [
            {"src": src, "mt": mt, "ref": ref}
            for mt in translations if mt.strip()
        ]

        # Calculate COMET-QE scores (without reference)
        comet_qe_items = [
            {"src": src, "mt": mt}
            for mt in translations if mt.strip()
        ]

        if comet_da_items:
            # Suppress COMET output
            old_stdout_fd = os.dup(1)
            old_stderr_fd = os.dup(2)
            devnull_fd = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull_fd, 1)
                os.dup2(devnull_fd, 2)
                da_scores_raw = da_model.predict(comet_da_items, batch_size=32, progress_bar=False)["scores"]
                qe_scores_raw = qe_model.predict(comet_qe_items, batch_size=32, progress_bar=False)["scores"]
            finally:
                os.dup2(old_stdout_fd, 1)
                os.dup2(old_stderr_fd, 2)
                os.close(devnull_fd)
                os.close(old_stdout_fd)
                os.close(old_stderr_fd)
                sys.stdout.flush()
                sys.stderr.flush()

            da_scores = [float(s) for s in da_scores_raw]
            qe_scores = [float(s) for s in qe_scores_raw]
        else:
            da_scores = [0.0] * len(translations)
            qe_scores = [0.0] * len(translations)

        # Store results
        result = {
            'src': src,
            'true_meaning': ref,
            'literal_translation': literal,
            'mt_1': translations[0] if len(translations) > 0 else "",
            'mt_2': translations[1] if len(translations) > 1 else "",
            'mt_3': translations[2] if len(translations) > 2 else "",
            'QE_mt_1': qe_scores[0] if len(qe_scores) > 0 else 0.0,
            'QE_mt_2': qe_scores[1] if len(qe_scores) > 1 else 0.0,
            'QE_mt_3': qe_scores[2] if len(qe_scores) > 2 else 0.0,
            'DA_mt_1': da_scores[0] if len(da_scores) > 0 else 0.0,
            'DA_mt_2': da_scores[1] if len(da_scores) > 1 else 0.0,
            'DA_mt_3': da_scores[2] if len(da_scores) > 2 else 0.0,
            'QE_mt_best': max(qe_scores) if qe_scores else 0.0,
            'QE_mt_avg': np.mean(qe_scores) if qe_scores else 0.0,
            'DA_mt_best': max(da_scores) if da_scores else 0.0,
            'DA_mt_avg': np.mean(da_scores) if da_scores else 0.0
        }
        results_list.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results_list)

    print(f"\n  → Translation complete!")
    print(f"  → Avg DA (best of {num_samples}): {results_df['DA_mt_best'].mean():.4f} ± {results_df['DA_mt_best'].std():.4f}")
    print(f"  → Avg DA (average): {results_df['DA_mt_avg'].mean():.4f} ± {results_df['DA_mt_avg'].std():.4f}")
    print(f"  → Avg QE (best of {num_samples}): {results_df['QE_mt_best'].mean():.4f} ± {results_df['QE_mt_best'].std():.4f}")
    print(f"  → Avg QE (average): {results_df['QE_mt_avg'].mean():.4f} ± {results_df['QE_mt_avg'].std():.4f}")

    # Clean up to free memory
    del model
    del tokenizer
    torch.cuda.empty_cache()

    return results_df

# ===== Step 3: Load Trained GRPO Models (placeholder) =====
# This will be replaced by loop below

# ===== Step 4: Load COMET Models (DA and QE) =====
print("\n[4/6] Loading COMET models (DA and QE)...")
# Suppress output during loading
import sys
old_stdout_fd = os.dup(1)
old_stderr_fd = os.dup(2)
devnull_fd = os.open(os.devnull, os.O_WRONLY)
try:
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    da_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da")).eval()
    qe_model = load_from_checkpoint(download_model("Unbabel/wmt20-comet-qe-da")).eval()
    if DEVICE == "cuda":
        try:
            da_model.to("cuda")
            qe_model.to("cuda")
        except Exception:
            pass
finally:
    os.dup2(old_stdout_fd, 1)
    os.dup2(old_stderr_fd, 2)
    os.close(devnull_fd)
    os.close(old_stdout_fd)
    os.close(old_stderr_fd)
    sys.stdout.flush()
    sys.stderr.flush()
print(f"  → COMET-DA model loaded")
print(f"  → COMET-QE model loaded")

# ===== Step 5: Evaluate All GRPO Models =====
print(f"\n[5/6] Evaluating all GRPO models on {len(test_df)} test samples...")
print(f"  → Generating {NUM_SAMPLES} samples per source")

def format_prompt(src_text, tokenizer):
    """Format prompt using the same template as training"""
    messages = [
        {"role": "system", "content": "You are a professional translator. Translate Chinese to natural English. Keep the meaning and tone."},
        {"role": "user", "content": f"Chinese: {src_text}\nEnglish:"}
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Evaluate each model
grpo_model_results = {}
for model_name, model_info in MODELS_TO_EVALUATE.items():
    results_df = evaluate_single_model(
        model_name=model_name,
        model_path=model_info['path'],
        test_df=test_df,
        da_model=da_model,
        qe_model=qe_model,
        num_samples=NUM_SAMPLES,
        skip_existing=SKIP_EXISTING_RESULTS
    )
    if results_df is not None:
        grpo_model_results[model_name] = {
            'df': results_df,
            'description': model_info['description'],
            'avg_da_best': float(results_df['DA_mt_best'].mean()),
            'std_da_best': float(results_df['DA_mt_best'].std()),
            'avg_da_mean': float(results_df['DA_mt_avg'].mean()),
            'std_da_mean': float(results_df['DA_mt_avg'].std()),
            'min_da': float(results_df['DA_mt_best'].min()),
            'max_da': float(results_df['DA_mt_best'].max()),
            'median_da': float(results_df['DA_mt_best'].median()),
            'avg_qe_best': float(results_df['QE_mt_best'].mean()),
            'std_qe_best': float(results_df['QE_mt_best'].std()),
            'avg_qe_mean': float(results_df['QE_mt_avg'].mean()),
            'std_qe_mean': float(results_df['QE_mt_avg'].std()),
            'min_qe': float(results_df['QE_mt_best'].min()),
            'max_qe': float(results_df['QE_mt_best'].max()),
            'median_qe': float(results_df['QE_mt_best'].median())
        }

if not grpo_model_results:
    print("\n⚠️  No models were successfully evaluated!")
    print("Please check that the model paths are correct.")
    exit(1)

print(f"\n  → All models evaluated successfully!")

# ===== Step 6: Compare Results and Save =====
print("\n[6/6] Comparing results and saving outputs...")

# Prepare comparison summary
comparison = {
    'evaluation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_set_size': int(len(test_df)),
    'num_samples_per_source': int(NUM_SAMPLES),
    'grpo_models': {},
    'baselines': {}
}

# Add GRPO model results
for model_name, model_data in grpo_model_results.items():
    comparison['grpo_models'][model_name] = {
        'description': model_data['description'],
        'avg_da_best': model_data['avg_da_best'],
        'std_da_best': model_data['std_da_best'],
        'avg_da_mean': model_data['avg_da_mean'],
        'std_da_mean': model_data['std_da_mean'],
        'min_da': model_data['min_da'],
        'max_da': model_data['max_da'],
        'median_da': model_data['median_da'],
        'avg_qe_best': model_data['avg_qe_best'],
        'std_qe_best': model_data['std_qe_best'],
        'avg_qe_mean': model_data['avg_qe_mean'],
        'std_qe_mean': model_data['std_qe_mean'],
        'min_qe': model_data['min_qe'],
        'max_qe': model_data['max_qe'],
        'median_qe': model_data['median_qe']
    }

# Add baseline comparisons
for baseline_name, baseline_data in baseline_results.items():
    baseline_entry = {
        'avg_da': float(baseline_data['avg_da']),
        'std_da': float(baseline_data['std_da']),
        'matched_samples': int(baseline_data['matched_count'])
    }
    # Add QE if available
    if 'avg_qe' in baseline_data:
        baseline_entry.update({
            'avg_qe': float(baseline_data['avg_qe']),
            'std_qe': float(baseline_data['std_qe'])
        })
    comparison['baselines'][baseline_name] = baseline_entry

# Save detailed results for each model
for model_name, model_data in grpo_model_results.items():
    results_path = os.path.join(OUTPUT_DIR, f"{model_name}_test_results.csv")
    model_data['df'].to_csv(results_path, index=False)
    print(f"  → {model_name} results saved to: {results_path}")

# Save comparison summary
comparison_path = os.path.join(OUTPUT_DIR, "comparison_summary.json")
with open(comparison_path, 'w') as f:
    json.dump(comparison, f, indent=2)
print(f"  → Comparison summary saved to: {comparison_path}")

# Print summary
print("\n" + "="*80)
print("EVALUATION SUMMARY")
print("="*80)
print(f"\nTest Set Size: {len(test_df)} samples")
print(f"Samples per source: {NUM_SAMPLES}")
print(f"\n{'Model':<30} {'Avg DA (Best)':<15} {'Std DA':<15} {'Description':<30}")
print("-"*95)

# Print GRPO models
for model_name, model_data in comparison['grpo_models'].items():
    avg_da = model_data['avg_da_best']
    std_da = model_data['std_da_best']
    desc = model_data['description']
    print(f"{model_name:<30} {avg_da:<15.4f} {std_da:<15.4f} {desc:<30}")

print("-"*95)

# Print baselines
for baseline_name, baseline_data in comparison['baselines'].items():
    avg_da = baseline_data['avg_da']
    std_da = baseline_data['std_da']
    matched = baseline_data['matched_samples']
    print(f"{baseline_name:<30} {avg_da:<15.4f} {std_da:<15.4f} {'Baseline (n=' + str(matched) + ')':<30}")

print("="*95)

# Additional statistics
print("\n" + "="*80)
print("DETAILED STATISTICS")
print("="*80)

for model_name, model_data in grpo_model_results.items():
    print(f"\n{model_name} ({model_data['description']}):")
    print(f"  DA Metrics:")
    print(f"    • Average (best of {NUM_SAMPLES}): {model_data['avg_da_best']:.4f} ± {model_data['std_da_best']:.4f}")
    print(f"    • Average (mean of {NUM_SAMPLES}): {model_data['avg_da_mean']:.4f} ± {model_data['std_da_mean']:.4f}")
    print(f"    • Median: {model_data['median_da']:.4f}")
    print(f"    • Range: [{model_data['min_da']:.4f}, {model_data['max_da']:.4f}]")
    print(f"  QE Metrics:")
    print(f"    • Average (best of {NUM_SAMPLES}): {model_data['avg_qe_best']:.4f} ± {model_data['std_qe_best']:.4f}")
    print(f"    • Average (mean of {NUM_SAMPLES}): {model_data['avg_qe_mean']:.4f} ± {model_data['std_qe_mean']:.4f}")
    print(f"    • Median: {model_data['median_qe']:.4f}")
    print(f"    • Range: [{model_data['min_qe']:.4f}, {model_data['max_qe']:.4f}]")

for baseline_name, baseline_data in baseline_results.items():
    print(f"\n{baseline_name}:")
    print(f"  Matched samples: {baseline_data['matched_count']}/{len(test_df)}")
    print(f"  DA Metrics:")
    print(f"    • Average: {baseline_data['avg_da']:.4f} ± {baseline_data['std_da']:.4f}")
    print(f"    • Median: {baseline_data['median_da']:.4f}")
    print(f"    • Range: [{baseline_data['min_da']:.4f}, {baseline_data['max_da']:.4f}]")
    if 'avg_qe' in baseline_data:
        print(f"  QE Metrics:")
        print(f"    • Average: {baseline_data['avg_qe']:.4f} ± {baseline_data['std_qe']:.4f}")
        print(f"    • Median: {baseline_data['median_qe']:.4f}")
        print(f"    • Range: [{baseline_data['min_qe']:.4f}, {baseline_data['max_qe']:.4f}]")

# Comparison with baselines
print("\n" + "="*80)
print("IMPROVEMENTS OVER BASELINES")
print("="*80)
for model_name, model_data in grpo_model_results.items():
    print(f"\n{model_name}:")
    for baseline_name, baseline_data in baseline_results.items():
        # DA improvement
        da_improvement = model_data['avg_da_best'] - baseline_data['avg_da']
        da_improvement_pct = (da_improvement / baseline_data['avg_da'] * 100) if baseline_data['avg_da'] > 0 else 0
        print(f"  vs {baseline_name} (DA): {da_improvement:+.4f} ({da_improvement_pct:+.1f}%)")

        # QE improvement (if available)
        if 'avg_qe' in baseline_data:
            qe_improvement = model_data['avg_qe_best'] - baseline_data['avg_qe']
            qe_improvement_pct = (qe_improvement / abs(baseline_data['avg_qe']) * 100) if baseline_data['avg_qe'] != 0 else 0
            print(f"  vs {baseline_name} (QE): {qe_improvement:+.4f} ({qe_improvement_pct:+.1f}%)")

print("\n" + "="*80)
print(f"All results saved to: {OUTPUT_DIR}")
print("="*80)

# Save sample translations for manual inspection (for each model)
sample_size = min(20, len(test_df))
for model_name, model_data in grpo_model_results.items():
    results_df = model_data['df']
    sample_df = results_df.head(sample_size)[['src', 'true_meaning', 'mt_1', 'DA_mt_1', 'DA_mt_best']]
    sample_path = os.path.join(OUTPUT_DIR, f"{model_name}_sample_translations.txt")
    with open(sample_path, 'w', encoding='utf-8') as f:
        f.write(f"Sample Translations - {model_name} (First {sample_size})\n")
        f.write("="*80 + "\n\n")
        for idx, row in sample_df.iterrows():
            f.write(f"Source: {row['src']}\n")
            f.write(f"Reference: {row['true_meaning']}\n")
            f.write(f"Translation: {row['mt_1']}\n")
            f.write(f"DA Score: {row['DA_mt_1']:.4f} (Best: {row['DA_mt_best']:.4f})\n")
            f.write("-"*80 + "\n\n")
    print(f"\nSample translations saved to: {sample_path}")

print("\nEvaluation complete! ✓")
