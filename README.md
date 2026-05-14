# LRPO

This is the official implementation for the paper: "Learning to Route Languages for Multilingual Preference Optimization".

[View the overview figure](figs/overview.jpeg)

LRPO is an online preference optimization method for multilingual LLMs, which treats the rollout language as a selectable training variable. For each question, it samples multilingual rollout groups, scores them with calibrated cross-lingual rewards, and updates both the policy model and a trainable language router.

LRPO builds on [`verl`](https://github.com/volcengine/verl), so most distributed training, rollout, checkpointing, and logging behavior follows the upstream `verl` interface.

## Method Summary

LRPO has three main pieces:

- **Language-routed rollouts:** for each training question, LRPO generates a group of responses in multiple target languages under a fixed rollout budget.
- **Calibrated multilingual rewards:** generated responses are compared with high-quality references using cross-lingual semantic similarity, then calibrated so scores are more comparable across language pairs.
- **Trainable language router:** a contextual multi-armed bandit learns topic- and region-conditioned language preferences and balances exploration with exploitation during training.


## Installation

Create a Python environment with CUDA-compatible PyTorch, then install the package in editable mode:

```bash
cd LRPO
pip install -e .
pip install -r requirements.txt
```

Optional rollout backends:

```bash
# vLLM backend
pip install ".[vllm]"

# SGLang backend
pip install ".[sglang]"
```

Reward code may require additional assets, depending on the reward function you use:

- a language identification model;
- mmBERT or other multilingual embedding models or multilingual reward models;
- offline calibration statistics for cross-lingual reward normalization.


## Data

LRPO uses the `verl` parquet format. Each row should provide the fields needed for prompting, reward computation, and language routing:

- `prompt`
- `reward_model.ground_truth`
- `ability`: topic/category label used by the language router
- `extra_info.language`: source or reference language for the example
- `extra_info.region`: region label for regional examples, if applicable

Preprocessing examples are available in `examples/data_preprocess/`. Treat them as templates and adapt them to your own dataset paths and schema.


## Training

The main training entry point is `verl.trainer.main_ppo` with LRPO-specific routing options:

```bash
python -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files=/path/to/train.parquet \
  data.val_files=/path/to/test.parquet \
  data.train_batch_size=2048 \
  data.max_prompt_length=512 \
  data.max_response_length=1024 \
  +data.dynamic_lang_policy=True \
  +data.lang_policy_alpha=0.1 \
  +data.lang_policy_update_every=5 \
  +data.lang_policy_temperature_init=1.0 \
  +data.lang_policy_temperature_min=0.3 \
  +data.lang_policy_temperature_decay=0.999 \
  +data.lang_policy_epsilon_init=0.2 \
  +data.lang_policy_epsilon_min=0.0 \
  +data.lang_policy_epsilon_decay=0.995 \
  +data.lang_policy_orig_lang_min=2 \
  +data.lang_policy_group_norm=zscore \
  actor_rollout_ref.model.path=/path/to/base-or-warm-start-model \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=8 \
  custom_reward_function.path=/path/to/reward_function.py \
  custom_reward_function.name=compute_score_batch \
  reward_model.reward_manager=batch \
  trainer.n_gpus_per_node=8 \
  trainer.total_epochs=4
```

See `examples/grpo_trainer/run_erm.sh` for a concrete launch script. Replace the data paths, checkpoint paths, reward-asset paths, logging paths, and `PYTHONPATH` before using it outside the original environment.

## Router Options

| Option | Meaning |
| --- | --- |
| `+data.dynamic_lang_policy` | Enables the online language router. |
| `+data.lang_policy_alpha` | Exponential moving average update rate for router values. |
| `+data.lang_policy_update_every` | Number of reward steps to buffer before updating the router. |
| `+data.lang_policy_temperature_init` | Initial softmax temperature for language sampling. |
| `+data.lang_policy_temperature_min` | Minimum annealed sampling temperature. |
| `+data.lang_policy_temperature_decay` | Temperature decay rate. |
| `+data.lang_policy_epsilon_init` | Initial epsilon-greedy exploration rate. |
| `+data.lang_policy_epsilon_min` | Minimum exploration rate. |
| `+data.lang_policy_epsilon_decay` | Exploration decay rate. |
| `+data.lang_policy_orig_lang_min` | Minimum number of original-language rollouts kept for each prompt group. |
| `+data.lang_policy_group_norm` | Reward normalization for router updates, for example `center` or `zscore`. |
| `+data.lang_policy_log_path` | Optional JSONL path for router probability logs. |


## Public Release Checklist

Before public reproduction, the repository still needs:

- add two calibration ways;
- released reward calibration files;
- removal or parameterization of private absolute paths.

## Citation

```bibtex
@misc{lrpo,
  title = {Learning to Route Languages for Multilingual Preference Optimization},
  author = {Geyang Guo and Hiromi Wakaki and Yuki Mitsufuji and Alan Ritter and Wei Xu},
  year = {2026},
  note = {ICML}
}
```
