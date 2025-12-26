from glob import glob
import pandas as pd

files = glob('results/*sft*.csv')
codes = list(set([f[:f.rfind("_")]for f in files]))

k = 1 # pass@k

for code in codes:
    result_files = glob(f'{code}*')
    results = {
        "qe": 0,
        "da": 0,
        "rouge": 0,
        "embed_distance": 0,
        "laj": 0
    }

    for i in range(k):
        df = pd.read_csv(f'{code}_{i}.csv', sep="|")

        # df = df[df['predicted'].notna()]
        # df = df[df['predicted'].apply(lambda x: len(x) > 1)]

        results['qe'] += df['qe'].mean()
        results['da'] += df['da'].mean()
        results['rouge'] += df['rouge'].mean()
        results['embed_distance'] += df['embed_distance'].mean()
        results['laj'] += df['laj'].mean()
    
    results['qe'] /= k
    results['da'] /= k
    results['rouge'] /= k
    results['embed_distance'] /= k
    results['laj'] /= k

    print(f'\n\n\n PASS @ {k} RESULTS for {code}')
    print("".join(["#"] * 150))

    da = round(results['da'] * 100, 2)
    qe = round(results['qe'] * 100, 2)
    rouge = round(results['rouge'] * 100, 2)
    embed_dist = round(results['embed_distance'] * 100, 2)
    laj = round(results['laj'], 2)

    print(f'& {da} & {qe} & {rouge} & {embed_dist} & {laj} \\\\')

    print("".join(["#"] * 150))
    print(f'PASS @ {k} RESULTS\n\n\n')
