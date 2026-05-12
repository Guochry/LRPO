# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections import defaultdict
from typing import Any, Optional

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.abstract import AbstractRewardManager


@register("naive")
class NaiveRewardManager(AbstractRewardManager):
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", alpha: float = 0.2, embedder_name: Optional[str] = "intfloat/multilingual-e5-large", clamp_sim: bool = True) -> None:
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

        self.alpha = float(alpha)
        self.clamp_sim = bool(clamp_sim)
        self.device = "cuda:0"
        self.embedder = None
        if embedder_name:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer(embedder_name, device=self.device)

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
                reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
                return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        B = len(data)
        base_reward = [0.0] * B
        sim_bonus = [0.0] * B
        texts = [""] * B
        uids = [None] * B
        last_pos = [0] * B

        already_print_data_sources = {}

        for i in range(B):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )

            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            # reward_tensor[i, valid_response_length - 1] = reward
            base_reward[i] = reward
            texts[i] = response_str
            uids[i] = data_item.non_tensor_batch.get("uid", None)
            last_pos[i] = valid_response_length - 1

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0
            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if self.embedder is not None:
            print("EMMM", self.device)
            with torch.no_grad():
                print("WOWWW")
                embs = self.embedder.encode(texts, convert_to_tensor=True, normalize_embeddings=True)  # [B, d]
                print("HEY")

        #     groups = defaultdict(list)
        #     for i, u in enumerate(uids):
        #         groups[u].append(i)

        #     for u, idxs in groups.items():
        #         # 找组内基础分最高的响应
        #         best_i = max(idxs, key=lambda i: base_reward[i])
        #         ref = embs[best_i].unsqueeze(0)
        #         sims = F.cosine_similarity(embs[idxs], ref, dim=-1)  # [m]
        #         for k, i in enumerate(idxs):
        #             s = sims[k].item()
        #             if self.clamp_sim:
        #                 s = max(0.0, min(1.0, s))  # [-1,1] → [0,1]
        #             sim_bonus[i] = self.alpha * s

        #     # 也可把统计信息塞进 extra
        #     reward_extra_info["sim_bonus"] = sim_bonus
        #     reward_extra_info["alpha"] = [self.alpha] * B

        # # —— 第三步：组合成最终奖励并回填到最后一个响应 token —— #
        # for i in range(B):
        #     final_r = base_reward[i] + sim_bonus[i]
        #     # 与官方 Naive 一致：只在“最后一个响应 token”上放句级奖励
        #     reward_tensor[i, last_pos[i]] = final_r

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor


# @register("naive")
# class NaiveRewardManager(AbstractRewardManager):
#     """The reward manager."""

#     def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source") -> None:
#         """
#         Initialize the NaiveRewardManager instance.

#         Args:
#             tokenizer: The tokenizer used to decode token IDs into text.
#             num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
#             compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
#             reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
#                 "data_source".
#         """
#         self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
#         self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
#         self.compute_score = compute_score or default_compute_score
#         self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source

#     def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
#         """We will expand this function gradually based on the available datasets"""

#         # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
#         if "rm_scores" in data.batch.keys():
#             if return_dict:
#                 reward_extra_keys = data.meta_info.get("reward_extra_keys", [])
#                 reward_extra_info = {key: data.non_tensor_batch[key] for key in reward_extra_keys}
#                 return {"reward_tensor": data.batch["rm_scores"], "reward_extra_info": reward_extra_info}
#             else:
#                 return data.batch["rm_scores"]

#         reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
#         reward_extra_info = defaultdict(list)

#         already_print_data_sources = {}

#         for i in range(len(data)):
#             data_item = data[i]  # DataProtoItem

#             prompt_ids = data_item.batch["prompts"]

#             prompt_length = prompt_ids.shape[-1]

#             valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
#             valid_prompt_ids = prompt_ids[-valid_prompt_length:]

#             response_ids = data_item.batch["responses"]
#             valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
#             valid_response_ids = response_ids[:valid_response_length]

#             # decode
#             prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
#             response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

#             ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
#             data_source = data_item.non_tensor_batch[self.reward_fn_key]
#             extra_info = data_item.non_tensor_batch.get("extra_info", {})
#             num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
#             extra_info["num_turns"] = num_turns

#             score = self.compute_score(
#                 data_source=data_source,
#                 solution_str=response_str,
#                 ground_truth=ground_truth,
#                 extra_info=extra_info,
#             )

#             if isinstance(score, dict):
#                 reward = score["score"]
#                 # Store the information including original reward
#                 for key, value in score.items():
#                     reward_extra_info[key].append(value)
#             else:
#                 reward = score

#             reward_tensor[i, valid_response_length - 1] = reward

#             if data_source not in already_print_data_sources:
#                 already_print_data_sources[data_source] = 0

#             if already_print_data_sources[data_source] < self.num_examine:
#                 already_print_data_sources[data_source] += 1
#                 print("[prompt]", prompt_str)
#                 print("[response]", response_str)
#                 print("[ground_truth]", ground_truth)
#                 if isinstance(score, dict):
#                     for key, value in score.items():
#                         print(f"[{key}]", value)
#                 else:
#                     print("[score]", score)

#         if return_dict:
#             return {
#                 "reward_tensor": reward_tensor,
#                 "reward_extra_info": reward_extra_info,
#             }
#         else:
#             return reward_tensor
