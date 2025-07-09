import os
import argparse

from huggingface_hub import snapshot_download

def download_model(repo_id: str, local_dir: str, revision: str = 'main'):
    """
    下载 HuggingFace 上的模型到指定目录。

    :param repo_id: 模型仓库标签（如 'Qwen/Qwen3-1.7B'）
    :param local_dir: 本地保存路径
    :param revision: 分支或版本，默认为 'main'
    """
    save_model_path = os.path.join(local_dir, repo_id.split('/')[-1])
    print(save_model_path)
    snapshot_download(
        repo_id=repo_id,
        local_dir=save_model_path,
        local_dir_use_symlinks=False,
        revision=revision
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, help="the model lable in huggingface")
    parser.add_argument("--save_path", type=str, help="the save model path")
    args = parser.parse_args()

    if not (
        args.save_path.startswith("/workspace/models/HF") or
        args.save_path.startswith("/workspace/models/Torch")
    ):
        raise ValueError("save_path 必须以 /workspace/models/HF 或 /workspace/models/Torch 开头")
    
    download_model(args.repo_id, args.save_path)