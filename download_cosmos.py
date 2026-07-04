import argparse
from huggingface_hub import login, snapshot_download
import os

parser = argparse.ArgumentParser(description="Download Cosmos models from HuggingFace")
parser.add_argument("--variant", type=str, default="DV8x8x8", help="Variant of the model to download")
parser.add_argument("--token", type=str, required=True, help="HuggingFace token")
args = parser.parse_args()

login(token=args.token, add_to_git_credential=True)
model_name = f"Cosmos-0.1-Tokenizer-{args.variant}"
hf_repo = "nvidia/" + model_name
local_dir = "pretrained_ckpts/" + model_name
os.makedirs(local_dir, exist_ok=True)
print(f"downloading {model_name}...")
snapshot_download(repo_id=hf_repo, local_dir=local_dir)