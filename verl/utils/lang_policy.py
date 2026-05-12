import math
from collections import defaultdict
from typing import Dict, Iterable, Tuple

import numpy as np


def parse_lang_mix(mix: str) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for seg in mix.split(","):
        lang, k = seg.split(":")
        weights[lang.strip()] = float(k)
    return weights


class LangBanditPolicy:
    def __init__(
        self,
        langs: Iterable[str],
        alpha: float = 0.1,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        prior_logits: Dict[str, np.ndarray] | None = None,
        region_prior_logits: Dict[str, np.ndarray] | None = None,
    ):
        self.langs = list(langs)
        self.lang_to_idx = {lang: idx for idx, lang in enumerate(self.langs)}
        self.alpha = alpha
        self.temperature = temperature
        self.epsilon = epsilon
        self._topic_q = defaultdict(lambda: np.zeros(len(self.langs), dtype=np.float32))
        self._region_q = defaultdict(lambda: np.zeros(len(self.langs), dtype=np.float32))
        if prior_logits:
            for topic, logits in prior_logits.items():
                self._topic_q[topic] = logits.astype(np.float32, copy=True)
        if region_prior_logits:
            for region, logits in region_prior_logits.items():
                self._region_q[region] = logits.astype(np.float32, copy=True)

    @classmethod
    def from_mix_strings(
        cls,
        topic_mix: Dict[str, str] | None,
        region_mix: Dict[str, str] | None,
        alpha: float = 0.1,
        temperature: float = 1.0,
        epsilon: float = 0.0,
        extra_langs: Iterable[str] | None = None,
    ) -> "LangBanditPolicy":
        langs = []
        topic_weights = {}
        region_weights = {}
        if topic_mix:
            for topic, mix in topic_mix.items():
                weights = parse_lang_mix(mix)
                topic_weights[topic] = weights
                langs.extend(weights.keys())
        if region_mix:
            for region, mix in region_mix.items():
                weights = parse_lang_mix(mix)
                region_weights[region] = weights
                langs.extend(weights.keys())

        if extra_langs:
            langs.extend([str(lang) for lang in extra_langs])

        langs = list(dict.fromkeys(langs))
        if not langs:
            langs = ["en"]

        def weights_to_logits(weights: Dict[str, float]) -> np.ndarray:
            # logits = np.zeros(len(langs), dtype=np.float32)
            logits = np.full(len(langs), math.log(1e-6), dtype=np.float32)
            for lang, weight in weights.items():
                idx = langs.index(lang)
                logits[idx] = math.log(max(weight, 1e-6))
            return logits

        topic_logits = {k: weights_to_logits(v) for k, v in topic_weights.items()}
        region_logits = {k: weights_to_logits(v) for k, v in region_weights.items()}
        # print("iiiiii", region_logits, "iiiiii", region_weights, "iiiiii")
        return cls(
            langs=langs,
            alpha=alpha,
            temperature=temperature,
            epsilon=epsilon,
            prior_logits=topic_logits,
            region_prior_logits=region_logits,
        )

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        scaled = logits / max(self.temperature, 1e-6)
        scaled -= scaled.max()
        exp = np.exp(scaled)
        return exp / exp.sum()

    def format_prob_matrix(self, max_keys: int | None = None) -> dict[str, dict[str, dict[str, float]]]:
        def probs_to_dict(probs: np.ndarray) -> dict[str, float]:
            return {lang: float(probs[idx]) for idx, lang in enumerate(self.langs)}

        def apply_epsilon(probs: np.ndarray) -> np.ndarray:
            if self.epsilon <= 0.0:
                return probs
            uniform = np.full_like(probs, 1.0 / len(probs))
            return (1.0 - self.epsilon) * probs + self.epsilon * uniform

        out: dict[str, dict[str, float]] = {}
        topic_keys = list(self._topic_q.keys())
        region_keys = list(self._region_q.keys())
        if max_keys is not None:
            topic_keys = topic_keys[:max_keys]
            region_keys = region_keys[:max_keys]
        for key in topic_keys:
            soft = self._softmax(self._topic_q[key])
            eff = apply_epsilon(soft)
            out[f"topic:{key}"] = {"softmax": probs_to_dict(soft), "effective": probs_to_dict(eff)}
        for key in region_keys:
            soft = self._softmax(self._region_q[key])
            eff = apply_epsilon(soft)
            out[f"region:{key}"] = {"softmax": probs_to_dict(soft), "effective": probs_to_dict(eff)}
        return out

    def _normalize_key_list(self, key) -> list[str]:
        if key is None:
            return []
        if isinstance(key, (list, tuple, np.ndarray)):
            return [str(item) for item in key if item is not None]
        return [str(key)]

    def sample(self, topic: str | None, region: str | None) -> Tuple[str, int, float]:
        logits = np.zeros(len(self.langs), dtype=np.float32)
        topic = self._normalize_key_list(topic)
        for tp in topic:
            logits = logits + self._topic_q[tp]
        if 'Regional Knowledge' in topic:
            regions = self._normalize_key_list(region)
            for rg in regions:
                logits = logits + self._region_q[rg]

        if self.epsilon > 0.0 and np.random.rand() < self.epsilon:
            idx = int(np.random.choice(len(self.langs)))
            return self.langs[idx], idx, 1.0 / len(self.langs)
        probs = self._softmax(logits)
        idx = int(np.random.choice(len(self.langs), p=probs))
        return self.langs[idx], idx, float(probs[idx])

    def update(self, topic: str | None, region: str | None, lang_idx: int, reward: float) -> None:
        topics = self._normalize_key_list(topic)
        regions = self._normalize_key_list(region)
        for tp in topics:
            current = self._topic_q[tp][lang_idx]
            self._topic_q[tp][lang_idx] = (1.0 - self.alpha) * current + self.alpha * reward
            print("update topic here: ", current, self._topic_q[tp][lang_idx], reward)
        if 'Regional Knowledge' in topics:
            for rg in regions:
                if rg not in self._region_q:
                    print("YOUUUUUUU", rg)
                current = self._region_q[rg][lang_idx]
                self._region_q[rg][lang_idx] = (1.0 - self.alpha) * current + self.alpha * reward
                print("update region here: ", current, self._region_q[rg][lang_idx], reward)

    def get_lang_idx(self, lang: str) -> int:
        return self.lang_to_idx[lang]
