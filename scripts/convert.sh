
python model_merger.py merge \
    --backend fsdp \
    --local_dir /srv/nlprx-lab/share6/gguo37/rl/dynamic_mrpo_router_tuning/examples/grpo_trainer/checkpoints/verl_grpo/warmstart-dynamic_router-quan-alpla0.1-update5-temperature0.3_1_0.999-epsilon_0.2_0_0.995-onpolicy2/global_step_64/actor \
    --hf_model_path Qwen/Qwen2.5-1.5B-Instruct \
    --target_dir /srv/nlprx-lab/share6/gguo37/rl/saved/warmstart-dynamic_router-quan-alpla0.1-update5-temperature0.3_1_0.999-epsilon_0.2_0_0.995-onpolicy2-64