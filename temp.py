from transformers import AutoConfig

config = AutoConfig.from_pretrained("/root/one_shot_rl/pretrained_checkpoints/Qwen-2-8b/checkpoints/Qwen2.5-1.5B")
print(config.model_type)  # should print 'qwen2'
