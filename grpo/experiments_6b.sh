export CUDA_VISIBLE_DEVICES=4,5,6,7
source run_grpo_verl.sh "Hindi" "Qwen/Qwen2.5-3B" "qwen3b-qe-cons" "reward_fn_qe.py" "compute_constrained"
source run_grpo_verl.sh "Hindi" "meta-llama/Llama-3.1-8B" "llama8b-qe-cons" "reward_fn_qe.py" "compute_constrained"
source run_grpo_verl.sh "Chinese" "Qwen/Qwen2.5-3B" "qwen3b-qe-cons" "reward_fn_qe.py" "compute_constrained"
source run_grpo_verl.sh "Chinese" "meta-llama/Llama-3.1-8B" "llama8b-qe-cons" "reward_fn_qe.py" "compute_constrained"