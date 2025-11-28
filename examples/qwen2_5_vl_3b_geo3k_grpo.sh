#!/bin/bash

set -x

export PYTHONUNBUFFERED=1


MODEL_PATH="/run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-VL-3B-Instruct"  # replace it with your local file path
MODEL_PATH="/run/determined/NAS1/public/HuggingFace/Qwen/Qwen2.5-VL-7B-Instruct"
MODEL_PATH="/home/chenhui/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7btool45grpo/global_step_40/actor/huggingface"
MODEL_PATH="/home/chenhui/EasyR1/checkpoints/easy_r1/qwen2_5_vl_3bgrpo/global_step_34/actor/huggingface"
MODEL_PATH="/home/chenhui/EasyR1/checkpoints/easy_r1/qwen2_5_vl_7b_grpo/global_step_40/actor/huggingface"
ray start --head --node-ip-address 0.0.0.0 --num-gpus 4 --ray-debugger-external --port 6379 

echo $CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=1,3,4,5
CUDA_VISIBLE_DEVICES=0,3,4,5 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/chenhui/EasyR1/train_1000grpo.json \
    data.val_files=/home/chenhui/EasyR1/val_1000grpo.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_grpo2 \
    trainer.n_gpus_per_node=4

ray start --head --node-ip-address 0.0.0.0 --num-gpus 4 --ray-debugger-external --port 6367     
CUDA_VISIBLE_DEVICES=2,3,4,5 python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files=/home/chenhui/EasyR1/AMG_train_tool1.json \
    data.val_files=/home/chenhui/EasyR1/AMG_test_tool1.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_7b_toolAMG2 \
    trainer.n_gpus_per_node=4