

python3 -m openrlhf.cli.train_ppo_ray \
--ref_num_nodes 1 \
--ref_num_gpus_per_node 1 \
--reward_num_nodes 1 \
--reward_num_gpus_per_node 1 \
--critic_num_nodes 1 \
--critic_num_gpus_per_node 1 \
--actor_num_nodes 1 \
--actor_num_gpus_per_node 1 \
--vllm_num_engines 1 \
--vllm_tensor_parallel_size 1 \
--vllm_gpu_memory_utilization 0.8 \
--advantage_estimator group_norm \
--pretrain /workspace/models/HF/Qwen3-0.6B \
--reward_pretrain /workspace/models/HF/Qwen3-0.6B \
--save_path /workspace/PipeRLHF/experiments/final/qwen3-06b-rlhf \
--ckpt_path /workspace/PipeRLHF/experiments/ckpt/qwen3-06b-rlhf \
--save_hf_ckpt \
--micro_train_batch_size 4 \
--train_batch_size 4 \
--micro_rollout_batch_size 4 \
--rollout_batch_size 2 \
--n_samples_per_prompt 2 \
--max_samples 2 \
--max_epochs 1 \
--prompt_max_len 1024 \
--generate_max_len 1024 \
--zero_stage 3 \
--bf16 \
--actor_learning_rate 5e-7 \
--critic_learning_rate 9e-6 \
--init_kl_coef 1e-3 \
--prompt_data /workspace/data/PipeRLHF/gsm8k/train.parquet \
--input_key prompt \
--apply_chat_template \
--normalize_reward \
--gradient_checkpointing \
--vllm_sync_backend nccl \
--enforce_eager \
--deepspeed_enable_sleep \
# --use_wandb c6cdaea552faf34cbfa1a02c8e055c715ebe6b7b \
# --wandb_org 1763777829-xxx \
# --wandb_project test_init \
# --wandb_run_name despreate_4_gpus_debug \