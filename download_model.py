#!/usr/bin/env python3

from huggingface_hub import snapshot_download

def download_model(repo_id, local_dir):
    print(f"Downloading {repo_id} to {local_dir}...")
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )
    print(f"Model downloaded successfully to {local_dir}")
source 
if __name__ == "__main__":
    repo_id = "Qwen/Qwen2.5-1.5B"
    local_dir = "./checkpoints/Qwen2.5-1.5B"
    download_model(repo_id, local_dir)
