from comet import download_model, load_from_checkpoint
import evaluate
import torch
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
import json
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE

def process(file):
    df = pd.read_csv(file, sep="|")
    source = list(df['src'])
    predicted = list(df['predicted'])
    ground_truth = list(df['true_meaning'])
    return source, predicted, ground_truth

def calculate_da(source, predicted, ground_truth):
    model_path = download_model("Unbabel/wmt22-comet-da")
    model = load_from_checkpoint(model_path)

    data = []
    for s, p, g in zip(source, predicted, ground_truth):
        data.append({
            "src": s,
            "mt": p,
            "ref": g
        })
    
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    del model
    return model_output['scores']

def calculate_qe(source, predicted, ground_truth):
    model_path = download_model("Unbabel/wmt22-cometkiwi-da")
    model = load_from_checkpoint(model_path)

    data = []
    for s, p, g in zip(source, predicted, ground_truth):
        data.append({
            "src": s,
            "mt": p
        })
    
    model_output = model.predict(data, batch_size=8, gpus=1)
    
    del model
    return model_output['scores']

def calculate_rouge(source, predicted, ground_truth):
    n = 3
    def char_ngrams(text, n):
        text = text.strip()
        if len(text) < n:
            return text
        return " ".join(text[i:i+n] for i in range(len(text) - n + 1))

    scorer = rouge_scorer.RougeScorer([f"rouge{n}"], use_stemmer=False)

    scores = []
    for pred, gt in zip(predicted, ground_truth):
        pred_ngrams = char_ngrams(pred, n)
        gt_ngrams = char_ngrams(gt, n)
        s = scorer.score(gt_ngrams, pred_ngrams)[f"rouge{n}"].fmeasure
        scores.append(s)

    del scorer
    return scores

def calculate_embed_distance(source, predicted, ground_truth):
    model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

    predicted_embeddings = model.encode(predicted)
    ground_truth_embeddings = model.encode(ground_truth)

    similarities = model.similarity(predicted_embeddings, ground_truth_embeddings)

    del model
    return similarities.diag().tolist()


def calculate_laj(source, predicted, ground_truth):
    # Absolute Grading: Outputs score of 1 to 5
    model = VLLM(model="prometheus-eval/prometheus-7b-v2.0", max_num_seqs=16)
    judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

    rubric_data = {
        "criteria": "Does the model output match the ground truth? NO translation is required. NO explanation is required. Brevity is NOT penalized. A perfect match must receive a 5/5. A literal translation should receive a 3/5, while a perfect semantic translation should receive a 5/5.",
        "score1_description": "The model output is empty.",
        "score2_description": "The model output contains a completely different sentence than the ground truth, without any overlap between the two responses.",
        "score3_description": "The model output contains a literal meaning of the ground truth response.",
        "score4_description": "The model output contains a semantically similar meaning of the ground truth response, but there are a few minor differences.",
        "score5_description": "The model output is the exact same as the ground truth response, except a few filler words that do not change the meaning of the phrase."
    }

    instructions = ["Given the following idiom, translate it semantically into English.\nIdiom: " + s for s in source]

    score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

    feedbacks, scores = judge.absolute_grade(
        instructions=instructions,
        responses=predicted,
        rubric=score_rubric,
        reference_answers=ground_truth
    )

    del model, judge
    return list(scores)

def compute_results(input_file, output_file):
    source, predicted, ground_truth = process(input_file)
    da = calculate_da(source, predicted, ground_truth)
    qe = calculate_qe(source, predicted, ground_truth)
    rouge = calculate_rouge(source, predicted, ground_truth)
    embed_distance = calculate_embed_distance(source, predicted, ground_truth)
    laj = calculate_laj(source, predicted, ground_truth)

    df = pd.DataFrame({
        "source": source,
        "predicted": predicted,
        "ground_truth": ground_truth,
        "da": da,
        "qe": qe,
        "rouge": rouge,
        "embed_distance": embed_distance,
        "laj": laj
    })

    df.to_csv(output_file, sep="|")

from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--file', type=str)
    args = parser.parse_args()

    file = args.file
    compute_results(file, file.replace('outputs', 'results'))
# from glob import glob
#     files = glob('outputs/*-outputs.csv')

#     for file in files:
#         compute_results(file, file.replace('outputs', 'results'))