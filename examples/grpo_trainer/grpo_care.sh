#!/bin/bash
#SBATCH --job-name=qwen3b_lrpo
#SBATCH --output=grpo.out
#SBATCH --error=grpo.err
#SBATCH --partition="nlprx-lab"
#SBATCH --exclude=xaea-12,dave,randotron,crushinator,shakey,baymax,chappie,trublu,deebot,megabot,samantha,chitti,nestor,kitt,tachikoma,uniblab
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node="l40s:8"
#SBATCH --cpus-per-task=96
#SBATCH --qos short

export BERTSCORE_LOG_PATH=/coc/pskynet6/gguo37/rl/log/run_bertscore_gt.tsv

source ~/.bashrc
conda activate RL
cd /coc/pskynet6/gguo37/rl/dynamic_mrpo_router_tuning/examples/grpo_trainer


bash run_erm.sh


