import os
import json
import requests
import numpy as np
from tqdm import tqdm
import torch


# tensorflow は GPT-2 公式重みの読み込みにしか使わないので、関数内で遅延 import する
# (pip install ".[gpt2]" を入れていない環境でも llm_from_scratch を import できるように)
def _import_tf():
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "GPT-2 の公式重みを読み込むには tensorflow が必要です: "
            'pip install "llm-from-scratch[gpt2]"'
        ) from e
    return tf


def download_and_load_gpt2(model_size, models_dir):
    tf = _import_tf()

    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    backup_base_url = "https://f001.backblazeb2.com/file/LLMs-from-scratch/gpt2"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)
        backup_url = os.path.join(backup_base_url, model_size, filename)
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path, backup_url)

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json"), "r", encoding="utf-8"))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    return settings, params


def download_file(url, destination, backup_url=None):
    def _attempt_download(download_url):
        response = requests.get(download_url, stream=True, timeout=60)
        response.raise_for_status()

        file_size = int(response.headers.get("Content-Length", 0))

        # Check if file exists and has same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size and file_size == file_size_local:
                print(f"File already exists and is up-to-date: {destination}")
                return True

        block_size = 1024  # 1 KB
        desc = os.path.basename(download_url)
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=desc) as progress_bar:
            with open(destination, "wb") as file:
                for chunk in response.iter_content(chunk_size=block_size):
                    if chunk:
                        file.write(chunk)
                        progress_bar.update(len(chunk))
        return True

    try:
        if _attempt_download(url):
            return
    except requests.exceptions.RequestException:
        if backup_url is not None:
            print(f"Primary URL ({url}) failed. Attempting backup URL: {backup_url}")
            try:
                if _attempt_download(backup_url):
                    return
            except requests.exceptions.RequestException:
                pass

        error_message = (
            f"Failed to download from both primary URL ({url})"
            f"{' and backup URL (' + backup_url + ')' if backup_url else ''}."
            "\nCheck your internet connection or the file availability.\n"
            "For help, visit: https://github.com/rasbt/LLMs-from-scratch/discussions/273"
        )
        print(error_message)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    tf = _import_tf()

    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

def load_weights_into_gpt(model, params):
    # Load weights for the embedding layer
    model.transformer.wte.weight.data = torch.tensor(params["wte"]["weight"], dtype=torch.float32)
    model.transformer.wpe.weight.data = torch.tensor(params["wpe"]["weight"], dtype=torch.float32)

    # Load weights for each transformer block
    for i, block in enumerate(model.transformer.blocks):
        block_params = params["blocks"][i]

        # Load attention weights and biases
        block.attn.c_attn.weight.data = torch.tensor(block_params["attn"]["c_attn"]["weight"], dtype=torch.float32)
        block.attn.c_attn.bias.data = torch.tensor(block_params["attn"]["c_attn"]["bias"], dtype=torch.float32)
        block.attn.c_proj.weight.data = torch.tensor(block_params["attn"]["c_proj"]["weight"], dtype=torch.float32)
        block.attn.c_proj.bias.data = torch.tensor(block_params["attn"]["c_proj"]["bias"], dtype=torch.float32)

        # Load layer normalization weights and biases
        block.ln_1.weight.data = torch.tensor(block_params["ln_1"]["weight"], dtype=torch.float32)
        block.ln_1.bias.data = torch.tensor(block_params["ln_1"]["bias"], dtype=torch.float32)
        block.ln_2.weight.data = torch.tensor(block_params["ln_2"]["weight"], dtype=torch.float32)
        block.ln_2.bias.data = torch.tensor(block_params["ln_2"]["bias"], dtype=torch.float32)

        # Load MLP weights and biases
        block.mlp.c_fc.weight.data = torch.tensor(block_params["mlp"]["c_fc"]["weight"], dtype=torch.float32)
        block.mlp.c_fc.bias.data = torch.tensor(block_params["mlp"]["c_fc"]["bias"], dtype=torch.float32)
        block.mlp.c_proj.weight.data = torch.tensor(block_params["mlp"]["c_proj"]["weight"], dtype=torch.float32)
        block.mlp.c_proj.bias.data = torch.tensor(block_params["mlp"]["c_proj"]["bias"], dtype=torch.float32)

    # Load final layer normalization weights and biases
    model.transformer.ln_f.weight.data = torch.tensor(params["ln_f"]["weight"], dtype=torch.float32)
    model.transformer.ln_f.bias.data = torch.tensor(params["ln_f"]["bias"], dtype=torch.float32)
