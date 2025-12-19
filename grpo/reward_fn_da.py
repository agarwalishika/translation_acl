import requests
import torch

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    SERVER_A = "http://172.22.224.17:5143/da-rewards"

    payload = {
        "completions": [solution_str],
        "true_meaning": [extra_info['true_meaning']],
        "literal_translation": [extra_info['literal_translation']],
    }

    r = requests.post(
        SERVER_A,
        json=payload,
        headers={"X-API-Key": ""},
        timeout=30,
    )
    # r.raise_for_status()
    print("status:", r.status_code)
    print("body:", r.text)
    r.raise_for_status()

    qe_rewards = r.json()["da_rewards"]
    return qe_rewards[0]