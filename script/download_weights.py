import argparse
import os
import shutil
from huggingface_hub import snapshot_download

available_models = [
    "marigold_appearance/finetuned",
    "marigold_appearance/pretrained",
    "marigold_lighting/finetuned",
    "marigold_lighting/pretrained",
    "rgbx/finetuned",
    "rgbx/pretrained"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="rgbx/finetuned",
        choices=available_models,
        help="Select model to download (default: rgbx/finetuned)"
    )
    parser.add_argument(
        "--local_dir",
        type=str,
        default="checkpoint",
        help="Directory to save the model"
    )

    args = parser.parse_args()

    HF_TOKEN = os.getenv("HF_TOKEN")
    LOCAL_DIR = args.local_dir
    selected_model = args.model

    if os.path.exists(LOCAL_DIR):
        if os.path.abspath(LOCAL_DIR) in ["/", os.path.expanduser("~")]:
            raise ValueError("Refusing to delete critical directory.")
        print(f"Removing existing directory: {LOCAL_DIR}")
        shutil.rmtree(LOCAL_DIR)

    print(f"Downloading model: {selected_model}")


    snapshot_download(
        repo_id="GDAOSU/olbedo",
        allow_patterns=f"{selected_model}/*",
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False,
        token=HF_TOKEN,
    )

    src = os.path.join(LOCAL_DIR, *selected_model.split("/"))

    for name in os.listdir(src):
        shutil.move(
            os.path.join(src, name),
            os.path.join(LOCAL_DIR, name)
        )

    top_level_folder = selected_model.split("/")[0]
    shutil.rmtree(os.path.join(LOCAL_DIR, top_level_folder), ignore_errors=True)
    shutil.rmtree(os.path.join(LOCAL_DIR, ".cache"), ignore_errors=True)

if __name__ == "__main__":
    main()
