"""
Utility functions for model quantization using bitsandbytes.
"""
import os
import gc
import torch
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
import bitsandbytes as bnb

def load_quantized_model(model_path, device_map="auto"):
    """
    Load a model with 8-bit quantization using bitsandbytes.
    
    Args:
        model_path: Path to the model
        device_map: Device mapping strategy (default: "auto" for automatic distribution)
        
    Returns:
        Quantized model
    """
    print(f"Loading quantized model from {model_path} with device_map={device_map}")
    
    # Force garbage collection to free up memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Configure quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    # Load model configuration
    config = AutoConfig.from_pretrained(model_path)
    
    # Load model with 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        quantization_config=quantization_config,
        device_map=device_map,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        offload_folder="offload_folder",
    )
    
    print(f"Model loaded with 8-bit quantization. Model type: {type(model)}")
    return model

def convert_linear_layers_to_8bit(model):
    """
    Convert linear layers in a model to 8-bit precision.
    
    Args:
        model: PyTorch model
        
    Returns:
        Model with 8-bit linear layers
    """
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            module = bnb.nn.Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=6.0,
            )
    return model

def optimize_memory():
    """
    Apply aggressive memory optimization techniques.
    """
    # Clear PyTorch cache
    torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    # Set memory efficient attention
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.2,max_split_size_mb:32"
    
    # Set PyTorch to release memory aggressively
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use at most 80% of available memory
    
    print("Applied aggressive memory optimizations")
