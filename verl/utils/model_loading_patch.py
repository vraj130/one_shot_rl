"""
Patch for model loading to enable 8-bit quantization.
"""
import os
import sys
import torch
from transformers import AutoConfig, AutoModelForCausalLM

# Add path to import our quantization module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from verl.utils.quantization import load_quantized_model

# Monkey patch the model loading function in VLLM
def patch_vllm_model_loading():
    """
    Apply monkey patches to VLLM model loading functions to enable 8-bit quantization.
    """
    try:
        # Import the module we want to patch
        from verl.third_party.vllm.vllm_v_0_6_3.model_loader import HFLoader
        
        # Store the original method
        original_load_model = HFLoader.load_model
        
        # Define our patched method
        def patched_load_model(self, actor_model, model_config, device_config, 
                              lora_config, parallel_config, scheduler_config, cache_config):
            """
            Patched version of load_model that uses 8-bit quantization.
            """
            print("Using patched model loading with 8-bit quantization")
            
            # If actor_model is already loaded, use it
            if hasattr(actor_model, 'parameters'):
                print("Actor model already loaded, using as is")
                return original_load_model(self, actor_model, model_config, device_config, 
                                         lora_config, parallel_config, scheduler_config, cache_config)
            
            # For multi-GPU, use "auto" device mapping
            device_map = "auto" if torch.cuda.device_count() > 1 else 0
            
            # Load model with quantization
            try:
                model_path = model_config.model
                print(f"Loading quantized model from {model_path}")
                quantized_model = load_quantized_model(model_path, device_map=device_map)
                
                # Use the quantized model
                return original_load_model(self, quantized_model, model_config, device_config, 
                                         lora_config, parallel_config, scheduler_config, cache_config)
            except Exception as e:
                print(f"Error loading quantized model: {e}")
                print("Falling back to original model loading")
                return original_load_model(self, actor_model, model_config, device_config, 
                                         lora_config, parallel_config, scheduler_config, cache_config)
        
        # Apply the patch
        HFLoader.load_model = patched_load_model
        print("Successfully patched VLLM model loading for 8-bit quantization")
        
        # Apply memory optimizations
        from verl.utils.quantization import optimize_memory
        optimize_memory()
        
    except ImportError as e:
        print(f"Could not patch VLLM model loading: {e}")
        print("Continuing without quantization")

if __name__ == "__main__":
    patch_vllm_model_loading()
