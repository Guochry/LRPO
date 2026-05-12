# Copyright 2025 Bytedance Ltd. and/or its affiliates
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

from concurrent.futures import ThreadPoolExecutor
from time import sleep

import requests, re, os

from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(
    "unsloth/gpt-oss-20b-BF16", trust_remote_code=True
)

MAX_TOTAL = 768      # prompt + solution 最多 token（给回复再留 ~70）

BASE_URL = os.getenv("GENRM_BASE_URL", "http://localhost:8000")
API_KEY = "EMPTY"
MODEL_NAME = "genrm-demo"
MAX_RETRIES = 3
BASE_DELAY = 2
MAX_WORKERS = 8

def build_prompt(problem: str, solution: str):
    with open('/srv/nlprx-lab/share6/gguo37/rl/fixed_mrpo/verl/utils/reward_score/total_prompt.txt', 'r', encoding='utf-8') as f:
        template = f.read()

    prompt = template.format(question=problem, response=solution)
    if len(TOKENIZER.encode(prompt, add_special_tokens=False)) <= MAX_TOTAL:
        return prompt, True

    tmp_prompt = template.format(question=problem, response="")
    head_len = len(TOKENIZER.encode(tmp_prompt, add_special_tokens=False))

    # 给 chat template 留点 buffer（可选但建议）
    overhead = 32
    budget = MAX_TOTAL - head_len - overhead
    if budget <= 0:
        return None, False

    sol_ids = TOKENIZER.encode(solution, add_special_tokens=False)
    sol_ids = sol_ids[:budget]  # ✅ 保留头部
    truncated_solution = TOKENIZER.decode(sol_ids, skip_special_tokens=True)

    prompt = template.format(question=problem, response=truncated_solution)

    # ✅ 最终再验一次，防止 decode/format 导致超标
    ids = TOKENIZER.encode(prompt, add_special_tokens=False)
    if len(ids) > MAX_TOTAL:
        ids = ids[:MAX_TOTAL]
        prompt = TOKENIZER.decode(ids, skip_special_tokens=True)

    return prompt, True


def get_response(problem, solution_str):

    prompt, keep = build_prompt(problem, solution_str)
    if not keep:
        return None
    messages = [{"role": "user", "content": prompt}]
    for attempt in range(MAX_RETRIES):
        try:
            headers = {"Content-Type": "application/json"}
            chat_url = f"{BASE_URL}/v1/chat/completions"
            data = {"model": MODEL_NAME, "messages": messages, "max_tokens": 1024, "temperature": 0.0, "reasoning_effort": "low"}
            output = requests.post(chat_url, headers=headers, json=data, timeout=(5, 120))
            print(output.json(), '~~~~~~~~~~')
            response = output.json()["choices"][0]["message"]["content"]
            return response
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print("Exception: ", repr(e))
                delay = BASE_DELAY * (2**attempt)
                print(f"Retrying in {delay} seconds...")
                sleep(delay)
            else:
                print(f"Failed after {MAX_RETRIES} attempts. Error: {e}")

    raise ConnectionRefusedError(f"Failed to run the model for {prompt}!")


def compute_reward(response):
    reward_score = 6.0
    match = re.search(r'\[\[(\d+)\]\]', response)
    if match:
        reward_score = int(match.group(1))  # 转换为整数
    else:
        match = re.search(r'Rating: (\d+)', response) 
        if match:
            reward_score = int(match.group(1))
        else: 
            match = re.findall(r'\d+', response)
            if match:
                reward_score = int(match[-1])
            else:
                print(response, "No match found")
    if reward_score>10:
        reward_score=10
    if reward_score<1:
        reward_score=1
    reward_score=reward_score/10
    return reward_score


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    try:
        problem = extra_info["question"] ###有时候这里会报错
    except:
        problem = "1+1=?"
    # print(extra_info)
    response = get_response(problem, solution_str)

    if response is not None:
        reward_score = compute_reward(response)
    else:
        reward_score = 6.0
    return reward_score


def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos):
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = []
        for data_source, solution_str, ground_truth, extra_info in zip(data_sources, solution_strs, ground_truths, extra_infos):
            future = executor.submit(compute_score, data_source, solution_str, ground_truth, extra_info)
            futures.append(future)

        results = [future.result() for future in futures]
        print('=====', results, '=====')

    return results
