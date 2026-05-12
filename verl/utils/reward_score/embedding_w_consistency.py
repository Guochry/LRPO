import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

import re
from functools import lru_cache
import fasttext


LID_MODEL_PATH = '/srv/nlprx-lab/share6/gguo37/rl/fixed_mrpo/examples/grpo_trainer/lid.176.ftz'
LID_DEVICE = "cpu"
ALLOWED_LANGS = {"ar","de","en","es","fr","id","it","ja","ko","nl","pl","pt","ru","vi","zh"}

FT_LANG_MAP = {
    "zh": "zh", "en": "en", "ar": "ar", "de": "de", "es": "es", "fr": "fr",
    "id": "id", "it": "it", "ja": "ja", "ko": "ko", "nl": "nl", "pl": "pl",
    "pt": "pt", "ru": "ru", "vi": "vi",
    # 如需扩展可以继续加，比如 "zh-cn": "zh"（不过 fastText 通常输出 zh）
}

def _normalize_text_for_lid(s: str) -> str:
    s = s.strip()
    s = re.sub(r"```.*?```", " ", s, flags=re.S)
    s = re.sub(r"\s+", " ", s)
    return s

def _is_text_too_weak_for_lid(s: str, min_chars: int = 1) -> bool:
    # 太短/几乎全是数字符号 -> LID 结果不可信
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


# ===================== E5 reward model =====================
E5_DEVICE = os.getenv("E5_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
E5_MAX_LEN = 512
E5_BATCH_SIZE = 128

E5_TOKENIZER = AutoTokenizer.from_pretrained('intfloat/multilingual-e5-small')
E5_MODEL = AutoModel.from_pretrained('intfloat/multilingual-e5-small').to(E5_DEVICE)
E5_MODEL.eval()

@torch.no_grad()
def _mean_pooling(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden_state: [B, L, H], attention_mask: [B, L]
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden_state)  # [B, L, 1]
    summed = (last_hidden_state * mask).sum(dim=1)                  # [B, L, 1] -> [B, H]
    counts = mask.sum(dim=1).clamp(min=1e-9)                        # [B, 1]
    return summed / counts

@torch.no_grad()
def encode_e5(texts, max_length=E5_MAX_LEN):
    batch = E5_TOKENIZER(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(E5_DEVICE)

    out = E5_MODEL(**batch)
    emb = _mean_pooling(out.last_hidden_state, batch["attention_mask"])
    emb = F.normalize(emb, p=2, dim=1)  # L2 normalize -> cosine = dot
    return emb  # [B, H]

@torch.no_grad()
def e5_cosine_sim(text_a_list, text_b_list):
    """
    返回每对 (a,b) 的 cosine similarity，shape [B]
    """
    assert len(text_a_list) == len(text_b_list)
    emb_a = encode_e5(text_a_list)
    emb_b = encode_e5(text_b_list)
    sim = (emb_a * emb_b).sum(dim=1)  # cosine since normalized
    return sim

def sim_to_reward(sim: torch.Tensor, tau: float = 0.5) -> torch.Tensor:
    r = (sim - tau) / (1.0 - tau)
    return torch.clamp(r, 0.0, 1.0)

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    if ground_truth is None:
        return 0.0
    if solution_str is None:
        return 0.0

    a = "query: " + str(solution_str).strip()
    b = "query: " + str(ground_truth).strip()

    sim = e5_cosine_sim([a], [b])[0].item()
    reward = sim_to_reward(torch.tensor([sim]))[0].item()
    return float(reward)


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    # ----------- E5 rewards -----------
    # print("OHHHHHHH")
    # print(len(data_sources), len(solution_strs))

    pairs = []
    for sol, gt in zip(solution_strs, ground_truths):
        sol = "" if sol is None else str(sol).strip()
        gt = "" if gt is None else str(gt).strip()
        pairs.append((sol, gt))

    e5_rewards = []
    bs = E5_BATCH_SIZE
    for i in range(0, len(pairs), bs):
        chunk = pairs[i:i+bs]
        a_list = ["query: " + s for s, _ in chunk]
        b_list = ["query: " + g for _, g in chunk]

        sim = e5_cosine_sim(a_list, b_list)          # tensor [B]
        r = sim_to_reward(sim).detach().cpu() # [B]
        e5_rewards.extend([float(x) for x in r.tolist()])

    # ----------- Language rewards -----------
    target_langs = [ds if isinstance(ds, str) else None for ds in data_sources]

    preds = predict_lang_fasttext([("" if s is None else str(s)) for s in solution_strs])
    lang_rewards = []
    for (pred_lang, prob), tgt in zip(preds, target_langs):
        lang_rewards.append(lang_match_reward(pred_lang, prob, tgt, soft=False))
    print(target_langs[:4], '======', preds[:4], '======', lang_rewards[:4])
    
    # ----------- Fuse -----------
    alpha = 0.7
    rewards = []
    embed_scores, consistency_scores = [], []
    for r_e5, r_lang in zip(e5_rewards, lang_rewards):
        rewards.append(float(alpha * r_e5 + (1.0 - alpha) * r_lang))
        embed_scores.append(r_e5)
        consistency_scores.append(r_lang)

    # print('=====', rewards, '=====')
    return rewards, embed_scores, consistency_scores
