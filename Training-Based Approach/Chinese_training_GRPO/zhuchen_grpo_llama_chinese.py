#!/usr/bin/env python3

import os, json, random, math
from datetime import datetime

# Disable warnings from tokenizers and Lightning
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
# Suppress all Lightning outputs and warnings
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["LIGHTNING_FABRIC_DISABLE_TIPS"] = "1"
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger("lightning").setLevel(logging.ERROR)
logging.getLogger("lightning_fabric").setLevel(logging.ERROR)
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM
from comet import download_model, load_from_checkpoint

import torch, gc

# 如果之前有 model / tokenizer / big_tensor 之类的，先 del 掉
for obj_name in ["model", "generator_model", "pipe"]:
    if obj_name in globals():
        del globals()[obj_name]

gc.collect()
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    torch.cuda.reset_peak_memory_stats()

# ----------------------------- Config -----------------------------
SEED = 1234
CSV_PATH = "petci_chinese_english_improved.csv"         # columns: src, true_meaning (literal optional)
MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"

# training hyperparams
NUM_EPOCHS = 3                  # 3-5 epochs recommended for 1100 samples
BATCH_SIZE = 2                  # Single GPU batch size
NUM_SAMPLES_PER_SRC = 4         # candidates per source (training)
LR = 5e-6                       # V1 settings - achieved 0.584 DA (vs 0.5486 baseline)
WD = 1e-2
GRAD_CLIP_NORM = 0.5
MAX_NEW_TOKENS = 96
GEN_TEMPERATURE = 0.8           # V1 settings
GEN_TOP_P = 0.9

# KL control - using absolute per-token diff
# V1 SETTINGS - despite mode collapse, achieved best DA score
KL_COEF_INIT = 0.05             # V1: Higher initial penalty
TARGET_KL = 2.0                 # V1: Lower target
KL_COEF_MIN = 0.001
KL_COEF_MAX = 0.5               # V1: Higher max (will hit ceiling, but DA improved)
REWARD_CLIP = 2.0               # V1: Standard clipping

# Saving
USE_DRIVE = False
DRIVE_DIR = "/content/drive/MyDrive/GRPO-Chinese-Eng/llama3.2-3b-grpo-comet-da-v1"  # used when USE_DRIVE=True
LOCAL_DIR = "/llama3.2-3b-grpo-comet-da-v1"

# ----------------------------- Repro & perf -----------------------------
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FORCE_BF16 = True

# Multi-GPU detection
NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 0
USE_MULTI_GPU = NUM_GPUS > 1

if DEVICE == "cuda":
    if FORCE_BF16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("Requested bfloat16 on a GPU without bf16 support.")
    DTYPE = torch.bfloat16 if FORCE_BF16 else torch.float16
else:
    DTYPE = torch.float32

print(f"[Info] Using device: {DEVICE} | dtype: {DTYPE}")
print(f"[Info] Available GPUs: {NUM_GPUS} | Multi-GPU: {USE_MULTI_GPU}")

# ----------------------------- (Optional) Mount Drive -----------------------------
if USE_DRIVE:
    try:
        from google.colab import drive  # type: ignore
        print("→ To save to Drive, run this once in a separate cell if not mounted yet:\nfrom google.colab import drive; drive.mount('/content/drive')")
        os.makedirs(DRIVE_DIR, exist_ok=True)
    except Exception:
        print("Google Colab not detected or drive unavailable; will save locally.")
        USE_DRIVE = False

# ----------------------------- Data -----------------------------
df = pd.read_csv(CSV_PATH)
for col in ["src", "true_meaning"]:
    if col not in df.columns:
        raise ValueError("CSV must contain ['src','true_meaning']")
df["src"] = df["src"].fillna("").astype(str)
df["true_meaning"] = df["true_meaning"].fillna("").astype(str)
df = df[df["src"].str.strip().str.len() > 0].reset_index(drop=True)

# 1123/500 split (1123 train, 500 test from 1623 total)
perm = np.random.RandomState(SEED).permutation(len(df))
mid = len(df) - 500  # 1123 train, 500 test
train_df = df.iloc[perm[:mid]].reset_index(drop=True)
test_df  = df.iloc[perm[mid:]].reset_index(drop=True)  # not used here, just held out

src_texts = train_df["src"].tolist()
references = train_df["true_meaning"].tolist()

print(f"[Info] Train size: {len(train_df)} | Test size: {len(test_df)}")

# ----------------------------- Models -----------------------------
tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
# Llama models need pad_token set
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
    tok.pad_token_id = tok.eos_token_id

# 加载模型（单GPU或多GPU）
print("Loading models...")
if USE_MULTI_GPU:
    # For DataParallel: load to CPU first, then move to GPU and wrap
    print(f"  → Using DataParallel across {NUM_GPUS} GPUs")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    )
    model = model.to("cuda:0")  # Move to primary GPU
    model = torch.nn.DataParallel(model, device_ids=list(range(NUM_GPUS)))
    model.train()

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=DTYPE, trust_remote_code=True
    )
    ref_model = ref_model.to("cuda:0")
    ref_model = torch.nn.DataParallel(ref_model, device_ids=list(range(NUM_GPUS)))
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
else:
    # Single GPU: use device_map="auto"
    print(f"  → Using single GPU")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=DTYPE, trust_remote_code=True
    )
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    ref_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, device_map="auto", torch_dtype=DTYPE, trust_remote_code=True
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

# ✅ 验证模型一致性
print("\nVerifying models are identical...")
test_input = torch.randint(0, 1000, (1, 10)).to("cuda:0" if DEVICE == "cuda" else "cpu")
with torch.no_grad():
    out1 = model(test_input).logits
    out2 = ref_model(test_input).logits
    max_diff = (out1 - out2).abs().max().item()

print(f"Max difference: {max_diff:.2e}")
if max_diff > 1e-4:
    print("🚨 ERROR: Models are NOT identical!")
    raise RuntimeError("Model initialization failed")
else:
    print("✅ Models are identical at initialization")

# reward model (COMET-DA)
# Suppress Lightning output during model loading at file descriptor level
import sys

old_stdout_fd = os.dup(1)
old_stderr_fd = os.dup(2)
devnull_fd = os.open(os.devnull, os.O_WRONLY)
try:
    os.dup2(devnull_fd, 1)
    os.dup2(devnull_fd, 2)
    da_model = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da")).eval()
    try:
        if DEVICE == "cuda":
            da_model.to("cuda")
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

def mdevice(m):
    """Get the device of a model (handles DataParallel)"""
    if isinstance(m, torch.nn.DataParallel):
        return next(m.module.parameters()).device
    return next(m.parameters()).device

# ----------------------------- GRPO Trainer -----------------------------
class GRPOTrainerDA:
    def __init__(self, model, ref_model, tokenizer, da_model, log_file=None):
        self.model = model
        self.ref_model = ref_model
        self.tok = tokenizer
        self.da = da_model
        self.kl_coef = KL_COEF_INIT
        self.target_kl = TARGET_KL
        self.kl_coef_min = KL_COEF_MIN
        self.kl_coef_max = KL_COEF_MAX
        self.last_avg_kl = 0.0

        # Running reward statistics for clipping
        self.reward_mean = 0.0
        self.reward_var = 1.0
        self.reward_count = 0

        # Training history for logging
        self.training_history = []
        self.global_step = 0

        # Real-time logging
        self.log_file = log_file
        if self.log_file:
            # Write CSV header
            with open(self.log_file, 'w') as f:
                f.write('global_step,epoch,loss,kl_coef,kl_penalty,reward_mean,reward_std,reward_min,reward_max,grad_norm,num_valid_groups\n')

    def _update_reward_stats(self, rewards_flat):
        """Update running mean and variance for reward normalization"""
        if len(rewards_flat) == 0:
            return
        batch_mean = np.mean(rewards_flat)
        batch_var = np.var(rewards_flat)
        batch_count = len(rewards_flat)

        # Welford's online algorithm
        delta = batch_mean - self.reward_mean
        self.reward_mean += delta * batch_count / (self.reward_count + batch_count)
        m_a = self.reward_var * self.reward_count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.reward_count * batch_count / (self.reward_count + batch_count)
        self.reward_var = M2 / (self.reward_count + batch_count)
        self.reward_count += batch_count

    def _clip_rewards(self, rewards):
        """Clip rewards to reduce outliers"""
        if self.reward_count < 10:  # Not enough data yet
            return rewards

        std = np.sqrt(self.reward_var + 1e-8)
        clipped = []
        for r_list in rewards:
            clipped_list = []
            for r in r_list:
                # Clip to [mean - REWARD_CLIP*std, mean + REWARD_CLIP*std]
                r_clipped = np.clip(r,
                                   self.reward_mean - REWARD_CLIP * std,
                                   self.reward_mean + REWARD_CLIP * std)
                clipped_list.append(r_clipped)
            clipped.append(clipped_list)
        return clipped

    def _prompt(self, src_text: str) -> str:
        # Llama 3.2 chat template
        messages = [
            {"role": "system", "content": "You are a professional translator. Translate Chinese to natural English. Keep the meaning and tone."},
            {"role": "user",   "content": f"Chinese: {src_text}\nEnglish:"}
        ]
        return self.tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate_batch(self, batch_src, num_samples=NUM_SAMPLES_PER_SRC):
        prompts = [self._prompt(s) for s in batch_src]
        enc = self.tok(prompts, return_tensors="pt", padding=True, truncation=True).to(mdevice(self.model))

        out = self.model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=True,
            temperature=GEN_TEMPERATURE,
            top_p=GEN_TOP_P,
            num_return_sequences=num_samples,
            pad_token_id=self.tok.eos_token_id
        )

        input_lens = (enc.input_ids != self.tok.pad_token_id).sum(dim=1).tolist()

        all_trans = []
        all_token_ids = []
        for i in range(len(batch_src)):
            group_texts = []
            group_token_ids = []
            actual_len = input_lens[i]

            for k in range(num_samples):
                idx = i * num_samples + k
                continuation = out[idx][actual_len:]
                if continuation.numel() == 0:
                    continuation = continuation.new_tensor([self.tok.eos_token_id])
                cont_ids = continuation.detach().clone()
                cont_ids_cpu = cont_ids.to("cpu")
                text = self.tok.decode(cont_ids_cpu, skip_special_tokens=True).strip()
                if text.lower().startswith("english:"):
                    text = text[len("english:"):].strip()
                group_texts.append(text)
                group_token_ids.append(cont_ids_cpu)

            all_trans.append(group_texts)
            all_token_ids.append(group_token_ids)

        return all_trans, all_token_ids

    @torch.no_grad()
    def comet_da_rewards(self, batch_src, batch_trans, batch_refs):
        import os
        import sys

        items, locs = [], []
        for i, (src, cands, ref) in enumerate(zip(batch_src, batch_trans, batch_refs)):
            for j, t in enumerate(cands):
                if isinstance(t, str) and t.strip():
                    items.append({"src": src, "mt": t, "ref": ref})
                    locs.append((i, j))
        scores = [0.0] * len(locs)
        if items:
            # Suppress Lightning output at file descriptor level
            old_stdout = os.dup(1)
            old_stderr = os.dup(2)
            devnull = os.open(os.devnull, os.O_WRONLY)
            try:
                os.dup2(devnull, 1)
                os.dup2(devnull, 2)
                sc = self.da.predict(items, batch_size=32, progress_bar=False)["scores"]
            finally:
                os.dup2(old_stdout, 1)
                os.dup2(old_stderr, 2)
                os.close(devnull)
                os.close(old_stdout)
                os.close(old_stderr)
                sys.stdout.flush()
                sys.stderr.flush()
            scores = [float(s) for s in sc]
        rewards = [[] for _ in range(len(batch_src))]
        for (i, j), s in zip(locs, scores):
            while len(rewards[i]) < j: rewards[i].append(0.0)
            rewards[i].append(s)
        for i in range(len(rewards)):
            while len(rewards[i]) < len(batch_trans[i]): rewards[i].append(0.0)
        return rewards

    def _seq_logprob(self, mdl, prompt_ids, cont_ids):
        """计算 log P(cont_ids | prompt_ids) 返回 (总和, 长度) 用于计算 per-token KL"""
        if cont_ids.numel() == 0:
            cont_ids = torch.tensor([self.tok.eos_token_id],
                                    device=prompt_ids.device,
                                    dtype=prompt_ids.dtype)

        prompt_ids = prompt_ids.detach()
        cont_ids = cont_ids.to(prompt_ids.device).detach()

        full_seq = torch.cat([prompt_ids, cont_ids], dim=0).unsqueeze(0)

        outputs = mdl(input_ids=full_seq)
        logits = outputs.logits[0]

        plen = len(prompt_ids)
        clen = len(cont_ids)

        cont_logits = logits[plen-1:plen+clen-1, :]
        log_probs = F.log_softmax(cont_logits, dim=-1)
        token_lps = log_probs.gather(-1, cont_ids.unsqueeze(-1)).squeeze(-1)

        return token_lps.sum(), clen

    def grpo_loss(self, batch_src, batch_trans, batch_cont_ids, batch_rewards):
        total, groups = 0.0, 0
        kls = []

        for src, cands, cont_ids_group, rs in zip(batch_src, batch_trans, batch_cont_ids, batch_rewards):
            triples = [
                (c, ids, r)
                for c, ids, r in zip(cands, cont_ids_group, rs)
                if isinstance(c, str) and c.strip()
            ]
            if len(triples) < 2:
                continue

            texts, cont_ids_list, rs = zip(*triples)
            r = torch.tensor(rs, device=mdevice(self.model), dtype=torch.float32)
            # Advantage normalization with more robust epsilon
            r_std = r.std()
            if r_std < 1e-4:  # All rewards are nearly identical
                A = torch.zeros_like(r)
            else:
                A = (r - r.mean()) / (r_std + 1e-8)

            prompt = self._prompt(src)
            prompt_ids = self.tok(prompt, return_tensors="pt").to(mdevice(self.model)).input_ids[0].detach()
            cont_ids_list = [ids.to(prompt_ids.device) for ids in cont_ids_list]

            lp_pi_list = []
            lp_ref_list = []
            lengths = []

            for cont_ids in cont_ids_list:
                lp_pi_sum, clen = self._seq_logprob(self.model, prompt_ids, cont_ids)
                lp_pi_list.append(lp_pi_sum)
                lengths.append(clen)

                with torch.no_grad():
                    lp_ref_sum, _ = self._seq_logprob(self.ref_model, prompt_ids, cont_ids)
                    lp_ref_list.append(lp_ref_sum)

            lp_pi = torch.stack(lp_pi_list)
            with torch.no_grad():
                lp_ref = torch.stack(lp_ref_list).detach()

            pg_loss = -(A.detach() * lp_pi).mean()

            # Per-token KL: divide by sequence length and use absolute value
            lengths_tensor = torch.tensor(lengths, device=lp_pi.device, dtype=torch.float32)
            per_token_kl_diff = (lp_pi - lp_ref) / (lengths_tensor + 1e-8)
            # Use absolute value for stable positive KL penalty
            approx_kl = per_token_kl_diff.abs().mean()
            kls.append(approx_kl.detach())

            loss = pg_loss + self.kl_coef * approx_kl
            total = total + loss
            groups += 1

        if groups > 0 and len(kls) > 0:
            mk = torch.stack(kls).mean().item()
            self.last_avg_kl = mk

            # Conservative KL adaptation with wider tolerance
            if mk > 1.5 * self.target_kl:
                self.kl_coef = min(self.kl_coef * 1.03, self.kl_coef_max)
            elif mk < 0.5 * self.target_kl:
                self.kl_coef = max(self.kl_coef * 0.97, self.kl_coef_min)

        if groups == 0:
            return torch.tensor(0.0, device=mdevice(self.model), requires_grad=True)

        return total / groups

    def train_step(self, batch_src, batch_refs, optimizer, epoch):
        self.model.train()

        translations, cont_token_ids = self.generate_batch(batch_src, NUM_SAMPLES_PER_SRC)
        rewards = self.comet_da_rewards(batch_src, translations, batch_refs)

        # Update reward statistics
        rewards_flat = [r for r_list in rewards for r in r_list if r != 0.0]
        self._update_reward_stats(rewards_flat)

        # Clip rewards to reduce variance
        rewards_clipped = self._clip_rewards(rewards)

        loss = self.grpo_loss(batch_src, translations, cont_token_ids, rewards_clipped)

        if not torch.isfinite(loss):
            print(f"  ⚠️ Invalid loss, skipping batch")
            return 0.0, {}

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), GRAD_CLIP_NORM)
        optimizer.step()

        # Collect metrics for logging
        self.global_step += 1
        metrics = {
            'global_step': self.global_step,
            'epoch': epoch,
            'loss': loss.item(),
            'kl_coef': self.kl_coef,
            'kl_penalty': self.last_avg_kl,
            'reward_mean': self.reward_mean,
            'reward_std': np.sqrt(self.reward_var),
            'reward_min': float(np.min(rewards_flat)) if rewards_flat else 0.0,
            'reward_max': float(np.max(rewards_flat)) if rewards_flat else 0.0,
            'grad_norm': float(grad_norm),
            'num_valid_groups': len(rewards_flat) // NUM_SAMPLES_PER_SRC if rewards_flat else 0
        }
        self.training_history.append(metrics)

        # Real-time logging: append to CSV file immediately
        if self.log_file:
            with open(self.log_file, 'a') as f:
                f.write(f"{metrics['global_step']},{metrics['epoch']},{metrics['loss']:.6f},"
                       f"{metrics['kl_coef']:.6f},{metrics['kl_penalty']:.6f},"
                       f"{metrics['reward_mean']:.6f},{metrics['reward_std']:.6f},"
                       f"{metrics['reward_min']:.6f},{metrics['reward_max']:.6f},"
                       f"{metrics['grad_norm']:.6f},{metrics['num_valid_groups']}\n")

        return loss.item(), metrics

# ----------------------------- Train -----------------------------
# Create log directory and real-time log file
SAVE_DIR = DRIVE_DIR if USE_DRIVE else LOCAL_DIR
os.makedirs(SAVE_DIR, exist_ok=True)
logs_dir = Path(SAVE_DIR) / "logs"
logs_dir.mkdir(parents=True, exist_ok=True)
realtime_log_file = logs_dir / "training_history_realtime.csv"

print(f"[Info] Real-time logs will be saved to: {realtime_log_file}")

trainer = GRPOTrainerDA(model, ref_model, tok, da_model, log_file=str(realtime_log_file))
optimizer = AdamW(model.parameters(), lr=LR, weight_decay=WD)

print("Starting GRPO training with COMET-DA (Llama 3.2 3B)…")
training_start_time = datetime.now()
print(f"Training started at: {training_start_time.strftime('%Y-%m-%d %H:%M:%S')}")

indices = np.arange(len(src_texts))
epoch_metrics = []  # Store epoch-level summaries

for epoch in range(NUM_EPOCHS):
    np.random.shuffle(indices)
    total_loss, n_batches = 0.0, 0
    epoch_start_time = datetime.now()
    print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")

    for i in range(0, len(indices), BATCH_SIZE):
        sel = indices[i:i+BATCH_SIZE]
        batch_src = [src_texts[j] for j in sel]
        batch_ref = [references[j] for j in sel]

        loss, metrics = trainer.train_step(batch_src, batch_ref, optimizer, epoch + 1)
        total_loss += loss; n_batches += 1

        if (i // BATCH_SIZE + 1) % 10 == 0:
            print(f"  Batch {i // BATCH_SIZE + 1}: loss={loss:.4f} | "
                  f"kl_coef={trainer.kl_coef:.5f} | "
                  f"kl_penalty={trainer.last_avg_kl:.6f} | "
                  f"reward_mean={trainer.reward_mean:.4f} | "
                  f"grad_norm={metrics.get('grad_norm', 0):.4f}")

    epoch_duration = (datetime.now() - epoch_start_time).total_seconds()
    avg_loss = total_loss / max(n_batches, 1)

    epoch_summary = {
        'epoch': epoch + 1,
        'avg_loss': avg_loss,
        'total_batches': n_batches,
        'duration_seconds': epoch_duration,
        'final_kl_coef': trainer.kl_coef,
        'final_reward_mean': trainer.reward_mean,
        'final_reward_std': np.sqrt(trainer.reward_var)
    }
    epoch_metrics.append(epoch_summary)

    print(f"Epoch {epoch+1} avg loss: {avg_loss:.4f} | duration: {epoch_duration:.1f}s")

training_duration = (datetime.now() - training_start_time).total_seconds()
print(f"\nGRPO training with COMET-DA completed!")
print(f"Total training time: {training_duration / 60:.2f} minutes")

# ----------------------------- Save -----------------------------
# SAVE_DIR and logs_dir already created before training

# Save model and tokenizer (unwrap DataParallel if needed)
if USE_MULTI_GPU:
    model.module.save_pretrained(SAVE_DIR)
else:
    model.save_pretrained(SAVE_DIR)
tok.save_pretrained(SAVE_DIR)
print(f"Saved model & tokenizer to: {SAVE_DIR}")

# Save data splits for reproducibility
split_dir = Path(SAVE_DIR) / "splits"
split_dir.mkdir(parents=True, exist_ok=True)
train_df.to_csv(split_dir / "train.csv", index=False)
test_df.to_csv(split_dir / "test.csv",  index=False)
print(f"Splits saved under: {split_dir}")

# ----------------------------- Save Training Logs -----------------------------
# 1. Save detailed batch-level training history (backup from memory)
history_df = pd.DataFrame(trainer.training_history)
history_csv_path = logs_dir / "training_history.csv"
history_df.to_csv(history_csv_path, index=False)
print(f"Training history (memory backup) saved to: {history_csv_path}")
print(f"Real-time log already saved to: {realtime_log_file}")

# 2. Save epoch-level summary
epoch_df = pd.DataFrame(epoch_metrics)
epoch_csv_path = logs_dir / "epoch_summary.csv"
epoch_df.to_csv(epoch_csv_path, index=False)
print(f"Epoch summary saved to: {epoch_csv_path}")

# 3. Save training configuration
config = {
    'model_id': MODEL_ID,
    'seed': SEED,
    'num_epochs': NUM_EPOCHS,
    'batch_size': BATCH_SIZE,
    'num_samples_per_src': NUM_SAMPLES_PER_SRC,
    'learning_rate': LR,
    'weight_decay': WD,
    'grad_clip_norm': GRAD_CLIP_NORM,
    'max_new_tokens': MAX_NEW_TOKENS,
    'gen_temperature': GEN_TEMPERATURE,
    'gen_top_p': GEN_TOP_P,
    'kl_coef_init': KL_COEF_INIT,
    'target_kl': TARGET_KL,
    'kl_coef_min': KL_COEF_MIN,
    'kl_coef_max': KL_COEF_MAX,
    'reward_clip': REWARD_CLIP,
    'train_size': len(train_df),
    'test_size': len(test_df),
    'training_start_time': training_start_time.strftime('%Y-%m-%d %H:%M:%S'),
    'training_duration_minutes': training_duration / 60,
    'device': str(DEVICE),
    'dtype': str(DTYPE),
    'num_gpus': NUM_GPUS,
    'multi_gpu': USE_MULTI_GPU
}
config_path = logs_dir / "config.json"
with open(config_path, 'w') as f:
    json.dump(config, f, indent=2)
print(f"Training config saved to: {config_path}")

# 4. Save final training statistics
final_stats = {
    'final_kl_coef': float(trainer.kl_coef),
    'final_reward_mean': float(trainer.reward_mean),
    'final_reward_std': float(np.sqrt(trainer.reward_var)),
    'total_steps': trainer.global_step,
    'training_duration_seconds': training_duration
}
stats_path = logs_dir / "final_stats.json"
with open(stats_path, 'w') as f:
    json.dump(final_stats, f, indent=2)
print(f"Final statistics saved to: {stats_path}")

print(f"\n{'='*60}")
print(f"All logs saved to: {logs_dir}")
print(f"{'='*60}")
print(f"\nTo analyze training, you can load:")
print(f"  - Batch metrics: {history_csv_path.name}")
print(f"  - Epoch summary: {epoch_csv_path.name}")
print(f"  - Configuration: {config_path.name}")
print(f"\nExample plotting code:")
print(f"  import pandas as pd")
print(f"  import matplotlib.pyplot as plt")
print(f"  df = pd.read_csv('{history_csv_path}')")
print(f"  plt.plot(df['global_step'], df['loss'])")
print(f"  plt.xlabel('Step'); plt.ylabel('Loss'); plt.show()")
