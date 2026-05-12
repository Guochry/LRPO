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
import json, bisect

CALIB_JSON = "/coc/pskynet5/gguo37/rl/reward_calibration/quantile/data/final_quantile.jsonl"
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

with open(CALIB_JSON, "r", encoding="utf-8") as f:
    _CAL = json.load(f)

PAIR_TABLES = _CAL["tables"]  # dict: "English|Chinese" -> table
def cdf_from_quantile_table(s: float, quantiles: list[float], qs: list[float]) -> float:
    if s <= quantiles[0]:
        return 0.0
    if s >= quantiles[-1]:
        return 1.0
    idx = bisect.bisect_left(quantiles, s)
    x0, x1 = quantiles[idx - 1], quantiles[idx]
    q0, q1 = qs[idx - 1], qs[idx]
    if x1 <= x0 + 1e-12:
        return float(q1)
    t = (s - x0) / (x1 - x0)
    return float(q0 + t * (q1 - q0))

def pair_key_from_codes(gt_lang_code: str | None, resp_lang_code: str | None) -> str | None:
    if gt_lang_code is None or resp_lang_code is None:
        return None
    a = gt_lang_code
    b = CODE_TO_NAME.get(resp_lang_code)
    if a is None or b is None:
        return None
    return "|".join(sorted([a, b]))

def calibrate_by_pair_cdf(raw_sim: float, gt_lang_code: str | None, resp_lang_code: str | None, use_ub_norm: bool = False) -> float:
    pk = pair_key_from_codes(gt_lang_code, resp_lang_code)
    if pk is None:
        return float(raw_sim)  # 或者返回一个默认值，比如 0.0/0.5，看你 reward 设计
    table = PAIR_TABLES.get(pk)
    if table is None:
        return float(raw_sim)

    s = float(raw_sim)

    # 可选：上界归一化（推荐，尤其同语 vs 跨语上界差异大）
    if use_ub_norm:
        U = float(table.get("ub_anchor", 1.0))
        s = min(1.0, s / (U + 1e-6))

    qs = table["qs"]
    mix_q = table["mix_quantiles"]
    u = cdf_from_quantile_table(s, mix_q, qs)  # 0~1
    return float(u)


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


def _normalize_gt_lang_for_quan(v):
    """
    Quantile calibration table keys use language names (e.g., 'English').
    Accept either lang code ('en') or language name.
    """
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    s_low = s.lower()
    if s_low in CODE_TO_NAME:
        return CODE_TO_NAME[s_low]
    return s.capitalize() if len(s.split()) == 1 else s


def _quan_reward_from_calibrated_value(v: float) -> float:
    """
    `calibrate_by_pair_cdf` usually returns a CDF value in [0,1].
    If calibration falls back to raw cosine, map it to [0,1].
    """
    x = float(v)
    if 0.0 <= x <= 1.0:
        return x
    return float(max(0.0, min(1.0, x)))


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    data_source: expected target lang code (e.g., 'zh','ar',...)
    We calibrate using (gt_lang = question language, resp_lang = predicted lang of solution).
    """
    if ground_truth is None or solution_str is None:
        return 0.0

    sol = str(solution_str).strip()
    gt = str(ground_truth).strip()

    raw_sim = float(mmbert_cosine_sim([sol], [gt])[0].item())

    pred_lang, _ = predict_lang_fasttext([sol])[0]

    # Prefer explicit question language from extra_info; otherwise fall back to data_source.
    gt_lang_src = extra_info.get("language") if isinstance(extra_info, dict) else data_source
    gt_lang = _normalize_gt_lang_for_quan(gt_lang_src)

    cal_val = calibrate_by_pair_cdf(
        raw_sim,
        gt_lang_code=gt_lang,
        resp_lang_code=pred_lang,
        use_ub_norm=False,
    )
    return _quan_reward_from_calibrated_value(cal_val)


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    # ----------- mmBERT embedding rewards (raw cosine) -----------
    pairs = []
    for sol, gt in zip(solution_strs, ground_truths):
        sol = "" if sol is None else str(sol).strip()
        gt = "" if gt is None else str(gt).strip()
        pairs.append((sol, gt))

    # LID for response language (for calibration + hard gate)
    preds = predict_lang_fasttext([("" if s is None else str(s)) for s in solution_strs])

    # target language codes (for hard gate)
    target_langs = [ds if isinstance(ds, str) else None for ds in data_sources]

    # question language for quantile calibration (expects names)
    gt_langs = []
    for ds, info in zip(data_sources, extra_infos):
        if isinstance(info, dict) and info.get("language") is not None:
            gt_langs.append(_normalize_gt_lang_for_quan(info.get("language")))
        else:
            gt_langs.append(_normalize_gt_lang_for_quan(ds))

    embed_rewards = []
    bs = MMBERT_BATCH_SIZE

    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i + bs]
        a_list = [s for s, _ in chunk]
        b_list = [g for _, g in chunk]

        raw_sims = mmbert_cosine_sim(a_list, b_list).detach().cpu().numpy().tolist()

        for k, s_raw in enumerate(raw_sims):
            global_idx = i + k
            gt_lang = gt_langs[global_idx]
            pred_lang, _ = preds[global_idx]

            cal_val = calibrate_by_pair_cdf(
                float(s_raw),
                gt_lang_code=gt_lang,
                resp_lang_code=pred_lang,
                use_ub_norm=True,
            )
            embed_rewards.append(_quan_reward_from_calibrated_value(cal_val))

    # ----------- Language rewards (unchanged) -----------
    lang_rewards = []
    for (pred_lang, prob), tgt in zip(preds, target_langs):
        lang_rewards.append(lang_match_reward(pred_lang, prob, tgt, soft=False))

    # ----------- Fuse (keep hard gate) -----------
    rewards = []
    consistency_scores = []
    for r_emb, r_lang in zip(embed_rewards, lang_rewards):
        rewards.append(float(r_emb) if r_lang == 1 else 0.0)
        consistency_scores.append(r_lang)

    return rewards, embed_rewards, consistency_scores
