set -x

export PYTHONPATH=/coc/pskynet6/gguo37/rl/dynamic_mrpo_router_tuning:$PYTHONPATH


# If you are using vllm<=0.6.3, you might need to set the following environment variable to avoid bugs:
# export VLLM_ATTENTION_BACKEND=XFORMERS

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/coc/pskynet6/gguo37/rl/dynamic_mrpo_router_tuning/examples/data_preprocess/data/care_helpsteer/train.parquet \
    data.val_files=/coc/pskynet6/gguo37/rl/dynamic_mrpo_router_tuning/examples/data_preprocess/data/care_helpsteer/test.parquet \
    data.train_batch_size=2048 \
    data.max_prompt_length=512 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.dynamic_lang_policy=True \
    \
    +data.lang_policy_alpha=0.1 \
    \
    +data.lang_policy_update_every=5 \
    \
    +data.lang_policy_temperature=1.0 \
    +data.lang_policy_temperature_init=1.0 \
    +data.lang_policy_temperature_min=0.3 \
    +data.lang_policy_temperature_decay=0.999 \
    \
    +data.lang_policy_epsilon_init=0.2 \
    +data.lang_policy_epsilon_min=0 \
    +data.lang_policy_epsilon_decay=0.995 \
    \
    +data.lang_policy_orig_lang_min=2 \
    +data.lang_policy_group_norm=zscore \
    \
    +data.lang_policy_log_every=1 \
    +data.lang_policy_log_max_keys=100 \
    +data.lang_policy_log_path=/coc/pskynet6/gguo37/rl/dynamic_mrpo_router_tuning/examples/grpo_trainer/router/qwen3b-warmstart-dynamic_router-quan-alpla0.1-update5-temperature0.3_1_0.999-epsilon_0.2_0_0.995-onpolicy2.jsonl \
    \
    actor_rollout_ref.model.path=/coc/pskynet6/gguo37/rl/saved/qwen2.5_3b_coldstart/checkpoint-44 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=256 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=40 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.log_val_generations=100 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_grpo' \
    trainer.experiment_name='qwen3b-warmstart-dynamic_router-quan-alpla0.1-update5-temperature0.3_1_0.999-epsilon_0.2_0_0.995-onpolicy2' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    custom_reward_function.path="/coc/pskynet6/gguo37/rl/dynamic_mrpo_router_tuning/verl/utils/reward_score/calibrated_bert_hard_fuse_quan.py" \
    custom_reward_function.name=compute_score_batch \
    reward_model.reward_manager=batch \
    trainer.total_epochs=4 $@