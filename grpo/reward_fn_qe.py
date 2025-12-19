import requests
import torch

def compute_pos(data_source, solution_str, ground_truth, extra_info=None):
    SERVER_A = "http://172.22.224.17:5142/qe-rewards"

    payload = {
        "completions": [solution_str],
        "mt": [extra_info['true_meaning']],
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

    qe_rewards = r.json()["qe_rewards"]
    return qe_rewards[0]

def compute_neg(data_source, solution_str, ground_truth, extra_info=None):
    SERVER_A = "http://172.22.224.17:5142/qe-rewards"

    payload = {
        "completions": [solution_str],
        "mt": [extra_info['literal_translation']],
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

    qe_rewards = -1 * r.json()["qe_rewards"]
    return qe_rewards[0]

def compute_constrained(data_source, solution_str, ground_truth, extra_info=None):
    return compute_pos(data_source, solution_str, ground_truth, extra_info) + compute_neg(data_source, solution_str, ground_truth, extra_info)