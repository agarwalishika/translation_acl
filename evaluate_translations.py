from comet import download_model, load_from_checkpoint
import evaluate
import torch
import pandas as pd
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

def process(file):
    df = pd.read_csv(file)
    source = list(df['source'])
    predicted = list(df['predicted'])
    ground_truth = list(df['ground_truth'])
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
    
    model_output = model.predict(data, batch_size=8, gpus=torch.cuda.device_count())
    
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
    
    model_output = model.predict(data, batch_size=8, gpus=torch.cuda.device_count())
    
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
    return [0] * len(predicted)

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

    df.to_csv(output_file)

compute_results('fake_input.csv', 'fake_results.csv')