import os
import csv
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import re
from functools import lru_cache
import fasttext
import numpy as np

# ===================== fastText LID (unchanged) =====================
LID_MODEL_PATH = '/coc/pskynet5/gguo37/rl/fixed_mrpo/examples/grpo_trainer/lid.176.ftz'
LID_DEVICE = "cpu"
ALLOWED_LANGS = {"ar","de","en","es","fr","id","it","ja","ko","nl","pl","pt","ru","vi","zh"}

FT_LANG_MAP = {
    "zh": "zh", "en": "en", "ar": "ar", "de": "de", "es": "es", "fr": "fr",
    "id": "id", "it": "it", "ja": "ja", "ko": "ko", "nl": "nl", "pl": "pl",
    "pt": "pt", "ru": "ru", "vi": "vi",
}

def _normalize_text_for_lid(s: str) -> str:
    s = s.strip()
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    return s

def _is_text_too_weak_for_lid(s: str, min_chars: int = 1) -> bool:
    if len(s) < min_chars:
        return True
    alnum = sum(ch.isalnum() for ch in s)
    if alnum / max(len(s), 1) < 0.05:
        return True
    return False

@lru_cache(maxsize=1)
def _get_lid_model():
    if fasttext is None:
        raise RuntimeError("fasttext not installed. Try: pip install fasttext-wheel")
    if not LID_MODEL_PATH or not os.path.exists(LID_MODEL_PATH):
        raise RuntimeError(
            "LID_MODEL_PATH not set or file not found. "
            "Download lid.176.ftz and set env LID_MODEL_PATH=/path/to/lid.176.ftz"
        )
    return fasttext.load_model(LID_MODEL_PATH)

def predict_lang_fasttext(texts):
    """
    return: list[(lang_code, prob)]
    """
    model = _get_lid_model()
    out = []
    for t in texts:
        t = _normalize_text_for_lid(t)
        if _is_text_too_weak_for_lid(t):
            out.append((None, 0.0))
            continue
        labels, probs = model.predict(t, k=1)  # labels like __label__en
        lang = labels[0].replace("__label__", "")
        prob = float(probs[0])
        lang = FT_LANG_MAP.get(lang, lang)
        if lang not in ALLOWED_LANGS:
            lang = None
            prob = 0.0
        out.append((lang, prob))
    return out

def lang_match_reward(pred_lang, pred_prob, target_lang, soft=True):
    """
    soft=True: 匹配按置信度给分；不匹配给 0
    soft=False: 匹配=1，不匹配=0
    """
    if target_lang is None:
        return 0.0
    if pred_lang is None:
        return 0.0
    if pred_lang != target_lang:
        return 0.0
    return float(pred_prob) if soft else 1.0


# ===================== Calibration stats (NEW) =====================
# Your stats file (unordered pairs)
STATS_CSV = "/coc/pskynet5/gguo37/rl/reward_calibration/sim_compute/data/unordered_pair_mean_std.csv"

# Anchor-to-reference calibration config (match your tested setting)
ALPHA_FIXED = None   # keep None to use adaptive alpha
K_SHRINK = 0
CALIB_CLIP_SIM = True  # clamp calibrated cosine to [-1,1] before mapping to reward

# Map fastText code -> CSV language name (must match csv exactly)
CODE_TO_NAME = {
    "zh": "Chinese",
    "ar": "Arabic",
    "en": "English",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ru": "Russian",
    "vi": "Vietnamese",
}

def _load_pair_stats(csv_path: str):
    """
    Expected header: lang_a, lang_b, count, mean, std
    Unordered key: tuple(sorted([a,b]))
    Returns: (stats_dict, mu_ref_weighted)
    """
    stats = {}
    means = []
    counts = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = row["lang_a"].strip()
            b = row["lang_b"].strip()
            n = int(row["count"])
            mu = float(row["mean"])
            sd = float(row["std"])
            key = tuple(sorted([a, b]))
            stats[key] = {"count": n, "mean": mu, "std": sd}
            means.append(mu)
            counts.append(n)

    if len(means) == 0:
        raise RuntimeError(f"No rows loaded from stats csv: {csv_path}")

    mu_ref = float(np.average(np.array(means), weights=np.array(counts)))
    return stats, mu_ref

PAIR_STATS, MU_REF = _load_pair_stats(STATS_CSV)

def _alpha_for_pair(n: int) -> float:
    if ALPHA_FIXED is not None:
        return float(ALPHA_FIXED)
    return float(n / (n + K_SHRINK))

def calibrate_anchor_to_ref_cosine(raw_sim: float, gt_lang_code: str | None, resp_lang_code: str | None) -> float:
    """
    raw_sim: cosine in [-1,1] (ideally)
    gt_lang_code/resp_lang_code: fastText lang codes like 'zh','ar',...
    Returns calibrated cosine (still roughly in [-1,1], we optionally clamp).
    """
    if gt_lang_code is None or resp_lang_code is None:
        return float(raw_sim)

    gt_name = gt_lang_code
    resp_name = CODE_TO_NAME.get(resp_lang_code)
    if gt_name is None or resp_name is None:
        return float(raw_sim)

    key = tuple(sorted([gt_name, resp_name]))
    row = PAIR_STATS.get(key)
    if row is None:
        return float(raw_sim)

    mu_pair = row["mean"]
    n = row["count"]
    alpha = _alpha_for_pair(n)

    cal = float(raw_sim - alpha * (mu_pair - MU_REF))
    if CALIB_CLIP_SIM:
        cal = float(max(-1.0, min(1.0, cal)))
    return cal


# ===================== mmBERT embedding reward (UPDATED) =====================
MMBERT_NAME = "jhu-clsp/mmBERT-small"
MMBERT_DEVICE = os.getenv("MMBERT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MMBERT_MAX_LEN = int(os.getenv("MMBERT_MAX_LEN", "512"))
MMBERT_BATCH_SIZE = int(os.getenv("MMBERT_BATCH_SIZE", "128"))

MMBERT_TOKENIZER = AutoTokenizer.from_pretrained(MMBERT_NAME)
MMBERT_MODEL = AutoModel.from_pretrained(MMBERT_NAME).to(MMBERT_DEVICE)
MMBERT_MODEL.eval()

@torch.no_grad()
def _masked_mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)   # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                         # [B, 1]
    return summed / counts

@torch.no_grad()
def encode_mmbert(texts, max_length=MMBERT_MAX_LEN) -> torch.Tensor:
    batch = MMBERT_TOKENIZER(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(MMBERT_DEVICE)

    out = MMBERT_MODEL(**batch)
    emb = _masked_mean_pooling(out.last_hidden_state, batch["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)
    return emb

@torch.no_grad()
def mmbert_cosine_sim(text_a_list, text_b_list) -> torch.Tensor:
    assert len(text_a_list) == len(text_b_list)
    emb_a = encode_mmbert(text_a_list)
    emb_b = encode_mmbert(text_b_list)
    return (emb_a * emb_b).sum(dim=1)  # [B]

def sim_to_reward(sim: torch.Tensor) -> torch.Tensor:
    # sim ∈ [-1, 1] -> r ∈ [0, 1]
    return torch.clamp((sim + 1.0) / 2.0, 0.0, 1.0)


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    data_source: expected target lang code (e.g., 'zh','ar',...)
    We calibrate using (gt_lang = data_source, resp_lang = predicted lang of solution).
    """
    if ground_truth is None or solution_str is None:
        return 0.0

    sol = str(solution_str).strip()
    gt = str(ground_truth).strip()

    # raw cosine
    raw_sim = float(mmbert_cosine_sim([sol], [gt])[0].item())

    # predicted response language (for calibration)
    (pred_lang, pred_prob) = predict_lang_fasttext([sol])[0]
    gt_lang = data_source if isinstance(data_source, str) else None

    # calibrated cosine
    cal_sim = calibrate_anchor_to_ref_cosine(raw_sim, gt_lang_code=gt_lang, resp_lang_code=pred_lang)

    # map to [0,1]
    reward = float(sim_to_reward(torch.tensor([cal_sim]))[0].item())
    return reward

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    # ----------- mmBERT embedding rewards (raw cosine) -----------
    pairs = []
    for sol, gt in zip(solution_strs, ground_truths):
        sol = "" if sol is None else str(sol).strip()
        gt = "" if gt is None else str(gt).strip()
        pairs.append((sol, gt))

    # LID for response language (for calibration + hard gate)
    preds = predict_lang_fasttext([("" if s is None else str(s)) for s in solution_strs])

    # target language codes
    target_langs = [ds if isinstance(ds, str) else None for ds in data_sources]
    gt_langs = [ds['language'].capitalize() for ds in extra_infos]

    embed_rewards = []
    bs = MMBERT_BATCH_SIZE

    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i+bs]
        a_list = [s for s, _ in chunk]
        b_list = [g for _, g in chunk]

        raw_sim = mmbert_cosine_sim(a_list, b_list).detach().cpu().numpy().tolist()  # list[float]

        # calibrate each example in this chunk
        for k, s_raw in enumerate(raw_sim):
            global_idx = i + k
            gt_lang = gt_langs[global_idx]
            pred_lang, _ = preds[global_idx]

            s_cal = calibrate_anchor_to_ref_cosine(float(s_raw), gt_lang_code=gt_lang, resp_lang_code=pred_lang)
            r_cal = float(sim_to_reward(torch.tensor([s_cal]))[0].item())

            embed_rewards.append(r_cal)

    # ----------- Language rewards (unchanged) -----------
    lang_rewards = []
    for (pred_lang, prob), tgt in zip(preds, target_langs):
        lang_rewards.append(lang_match_reward(pred_lang, prob, tgt, soft=False))

    print(target_langs[:10], '======', preds[:10], '======', lang_rewards[:10], '======', solution_strs[:10], '======', ground_truths[:10])

    # ----------- Fuse (keep your hard gate) -----------
    rewards = []
    consistency_scores = []
    for r_emb, r_lang in zip(embed_rewards, lang_rewards):
        rewards.append(float(r_emb) if r_lang == 1 else 0.0)
        consistency_scores.append(r_lang)

    return rewards, embed_rewards, consistency_scores
