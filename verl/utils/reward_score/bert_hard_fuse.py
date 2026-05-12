import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import re
from functools import lru_cache
import fasttext


# ===================== fastText LID (unchanged) =====================
LID_MODEL_PATH = '/srv/nlprx-lab/share5/gguo37/rl/fixed_mrpo/examples/grpo_trainer/lid.176.ftz'
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


# ===================== mmBERT embedding reward (NEW) =====================
MMBERT_NAME = "jhu-clsp/mmBERT-small"
MMBERT_DEVICE = os.getenv("MMBERT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MMBERT_MAX_LEN = int(os.getenv("MMBERT_MAX_LEN", "512"))
MMBERT_BATCH_SIZE = int(os.getenv("MMBERT_BATCH_SIZE", "128"))

MMBERT_TOKENIZER = AutoTokenizer.from_pretrained(MMBERT_NAME)
MMBERT_MODEL = AutoModel.from_pretrained(MMBERT_NAME).to(MMBERT_DEVICE)
MMBERT_MODEL.eval()

@torch.no_grad()
def _masked_mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    last_hidden_state: [B, L, H]
    attention_mask:   [B, L]
    """
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)   # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                   # [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                         # [B, 1]
    return summed / counts

@torch.no_grad()
def encode_mmbert(texts, max_length=MMBERT_MAX_LEN) -> torch.Tensor:
    """
    Returns L2-normalized embeddings: [B, H]
    Note: mmBERT model card's example uses mean over tokens; we use attention-mask mean pooling
    to avoid padding affecting embeddings. :contentReference[oaicite:1]{index=1}
    """
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
    """
    cosine similarity for each pair (a,b), shape [B]
    """
    assert len(text_a_list) == len(text_b_list)
    emb_a = encode_mmbert(text_a_list)
    emb_b = encode_mmbert(text_b_list)
    return (emb_a * emb_b).sum(dim=1)


def sim_to_reward(sim):
    # sim ∈ [-1, 1] -> r ∈ [0, 1]
    return torch.clamp((sim + 1.0) / 2.0, 0.0, 1.0)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if ground_truth is None or solution_str is None:
        return 0.0

    # mmBERT usage example in model card feeds raw text (no "query:" prefix). :contentReference[oaicite:2]{index=2}
    a = str(solution_str).strip()
    b = str(ground_truth).strip()

    sim = mmbert_cosine_sim([a], [b])[0].item()
    reward = sim_to_reward(torch.tensor([sim]))[0].item()
    return float(reward)

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    # ----------- mmBERT embedding rewards -----------
    pairs = []
    for sol, gt in zip(solution_strs, ground_truths):
        sol = "" if sol is None else str(sol).strip()
        gt = "" if gt is None else str(gt).strip()
        pairs.append((sol, gt))

    embed_rewards = []
    bs = MMBERT_BATCH_SIZE
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i+bs]
        a_list = [s for s, _ in chunk]
        b_list = [g for _, g in chunk]

        sim = mmbert_cosine_sim(a_list, b_list)            # tensor [B]
        r = sim_to_reward(sim).detach().cpu()              # [B]
        embed_rewards.extend([float(x) for x in r.tolist()])

    # ----------- Language rewards (unchanged) -----------
    target_langs = [ds if isinstance(ds, str) else None for ds in data_sources]
    preds = predict_lang_fasttext([("" if s is None else str(s)) for s in solution_strs])

    lang_rewards = []
    for (pred_lang, prob), tgt in zip(preds, target_langs):
        lang_rewards.append(lang_match_reward(pred_lang, prob, tgt, soft=False))

    print(target_langs[:10], '======', preds[:10], '======', lang_rewards[:10], '======', solution_strs[:10], '======', ground_truths[:10])

    # ----------- Fuse (keep your hard gate) -----------
    rewards = []
    embed_scores, consistency_scores = [], []
    for r_emb, r_lang in zip(embed_rewards, lang_rewards):
        if r_lang == 1:
            rewards.append(float(r_emb))
        else:
            rewards.append(0.0)
        embed_scores.append(r_emb)
        consistency_scores.append(r_lang)

    # print('=====', rewards, '=====')
    return rewards, embed_scores, consistency_scores
