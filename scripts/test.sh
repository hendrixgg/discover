#!/bin/sh

# Force JAX to use the libraries in your virtual environment
# Explicitly include nvjitlink, which is required for JAX + CUDA 12
export VENV_LIB=$HOME/discover/.venv/lib/python3.10/site-packages
export LD_LIBRARY_PATH=$VENV_LIB/nvidia/nvjitlink/lib:$VENV_LIB/nvidia/cudnn/lib:$VENV_LIB/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH

# Debugging: Tell JAX to be loud about why it fails
export TF_CPP_MIN_LOG_LEVEL=0

# JAX Memory & GPU Settings for V100s
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export CUDA_VISIBLE_DEVICES=0  # Use only the first GPU for now to test
export TF_FORCE_GPU_ALLOW_GROWTH=true

POINT_MAZE_SIMPLE="--env_name=high_dimensional_maze --env_file_name=../maze_layouts/rand_2d_10h.pkl --num_timesteps=100000 --episode_length=100 --num_envs=1 --train_step_multiplier=1 --num_goals=100 --unroll_length=100 --batch_size=1"

DISCOVER="--algo=TD3 --goal_selection_strategy=UCB --ucb_std_tar=0.0 --ucb_std_ini=1.0 --ucb_mean_tar=1.0 --ucb_mean_ini=0.0 --filter_goals=none --relabel_prob_future=0.7"

EXP_NAME="test"
wandb_project_name="test"
wandb_group_name="test"
ENV=$POINT_MAZE_SIMPLE
ALGO=$DISCOVER

echo "Start Training."
python ../training.py  $ENV $ALGO --project_name=$wandb_project_name --group_name=$wandb_group_name --exp_name=$EXP_NAME \
    --activate_final --tau_policy=5e-07 --critic_lr=0.0003 --deterministic_eval  --transform_ucb --ensemble_size=6 \
    --goal_log_delay=5 --render_delay=2  --seed=0  --adaptation_strategy=simple --log_wandb  
echo "Finished Training."
