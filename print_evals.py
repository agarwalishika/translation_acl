from glob import glob
files = glob('outputs/*-outputs*.csv')
import os

for i in range(0, len(files), 1):
    # num = int(files[i].split("_")[-1].split('.')[0])
    # if num > 1 or "sft" not in files[i]: continue
    # if "nllb" in files[i] or "grpo" in files[i]: continue
    with open(f'run_evals_0.sh', 'a+') as f:
        f.write(f"py evaluate_translations.py --file \"{files[i]}\"\n")

    # num = int(files[i+1].split("_")[-1].split('.')[0])
    # if num > 7 or "grpo" in files[i]: continue
    # with open(f'run_evals_1.sh', 'a+') as f:
    #     f.write(f"py evaluate_translations.py --file \"{files[i+1]}\"\n")
    # with open(f'run_evals_2.sh', 'a+') as f:
    #     f.write(f"py evaluate_translations.py --file \"{files[i+2]}\"\n")
    # with open(f'run_evals_3.sh', 'a+') as f:
    #     f.write(f"py evaluate_translations.py --file \"{files[i+3]}\"\n")