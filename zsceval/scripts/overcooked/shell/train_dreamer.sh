
#!/bin/bash
env="Overcooked"

layout=$1

if [[ "${layout}" == "random0" || "${layout}" == "random0_medium" || "${layout}" == "random1" || "${layout}" == "random3" || "${layout}" == "small_corridor" || "${layout}" == "unident_s" ]]; then
    version="old"
else
    version="new"
fi

entropy_coefs="0.2 0.05 0.01"
entropy_coef_horizons="0 2.5e6 5e6"
if [[ "${layout}" == "small_corridor" ]]; then
    entropy_coefs="0.2 0.05 0.01"
    entropy_coef_horizons="0 2.5e6 5e6"
fi
reward_shaping_horizon="5e7"
num_env_steps="1e6"

num_agents=2
population_size=12
algo="dreamer_adaptive"
exp="dreamer-s${population_size}"
stage="S2"
seed_begin=1
seed_max=3
path=../../policy_pool

export POLICY_POOL=${path}

n_training_threads=50

ulimit -n 65536

echo "env is ${env}, layout is ${layout}, algo is ${algo}, exp is ${exp},  from ${seed_begin} to ${seed_max}, stage is ${stage}"
for seed in $(seq ${seed_begin} ${seed_max});
# for seed in 1 2 5;
do
    python train/train_adaptive_dreamer.py --env_name ${env} --algorithm_name ${algo} --experiment_name "${exp}" --layout_name ${layout} --num_agents ${num_agents} \
    --seed ${seed} --n_training_threads 1 --num_mini_batch 36 --episode_length 400 --num_env_steps ${num_env_steps} --reward_shaping_horizon ${reward_shaping_horizon} \
    --overcooked_version ${version} \
    --n_rollout_threads ${n_training_threads} --dummy_batch_size 1 \
    --ppo_epoch 15 --entropy_coefs ${entropy_coefs} --entropy_coef_horizons ${entropy_coef_horizons} \
    --stage 2 \
    --save_interval 25 --log_interval 1  --eval_interval 5 --n_eval_rollout_threads $((population_size * 2)) --eval_episodes 5 \
    --population_yaml_path ${path}/${layout}/dreamer/train-s${population_size}-dreamer-${seed}.yml \
    --population_size ${population_size} --adaptive_agent_name dreamer --use_agent_policy_id \
    --use_proper_time_limits --wandb_name "your_entity_name" --use_eval
done
