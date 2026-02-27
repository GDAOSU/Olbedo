# Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments

[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://gdaosu.github.io/olbedo/)
[![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-orange)](https://huggingface.co/spaces/GDAOSU/olbedo)
[![Hugging Face Models](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-yellow)](https://huggingface.co/GDAOSU/olbedo)
[![Hugging Face Datasets](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-green)](https://huggingface.co/datasets/GDAOSU/Olbedo)
[![arXiv](https://img.shields.io/badge/arXiv-2602.22025-b31b1b.svg)](https://arxiv.org/abs/2602.22025)

This repository contains the official implementation and inference code for **Olbedo**.

## 🔗 Resources

We provide a comprehensive suite of resources for this project:

* **Project Page:** [https://gdaosu.github.io/olbedo/](https://gdaosu.github.io/olbedo/)
* **Interactive Demo:** [Hugging Face Spaces](https://huggingface.co/spaces/GDAOSU/olbedo)
* **Pre-trained Models:** [Hugging Face Model Hub](https://huggingface.co/GDAOSU/olbedo)
* **Dataset:** [Hugging Face Datasets](https://huggingface.co/datasets/GDAOSU/Olbedo)
* **Paper:** [arXiv:2602.22025](https://arxiv.org/abs/2602.22025)

## 🚀 Usage

We provide Docker support to ensure a consistent environment for running inference.

### 1. Build the Environment

First, clone this repository and build the Docker image. This will set up all necessary dependencies.

```bash
bash build_docker.sh
```

### 2. Run Inference

To run inference on your own images, use the run_inference.sh script. You must specify the input directory containing your images and the output directory where results will be saved.

```bash
bash run_inference.sh <input_directory> <output_directory>
```

## 📂 Data & Models

If you wish to use the data or models separately, they are hosted on Hugging Face:

| Resource | Link | Description |
| :--- | :--- | :--- |
| **Model Weights** | [Download Here](https://huggingface.co/GDAOSU/olbedo) | Pre-trained checkpoints for the Olbedo architecture. |
| **Dataset** | [Download Here](https://huggingface.co/datasets/GDAOSU/Olbedo) | The dataset used for training and evaluation. |


## 📝 Citation

If you find this project useful for your research, please consider citing our work:

```
@misc{song2026olbedoalbedoshadingaerial,
      title={Olbedo: An Albedo and Shading Aerial Dataset for Large-Scale Outdoor Environments}, 
      author={Shuang Song and Debao Huang and Deyan Deng and Haolin Xiong and Yang Tang and Yajie Zhao and Rongjun Qin},
      year={2026},
      eprint={2602.22025},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={[https://arxiv.org/abs/2602.22025](https://arxiv.org/abs/2602.22025)}, 
}
```

## 🙏 Acknowledgements

This codebase is largely built upon the following excellent projects:

* **[Marigold](https://github.com/prs-eth/Marigold)**
* **[RGBX](https://github.com/zheng95z/rgbx)**

We thank the authors for their open-source contributions.

## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The Marigold pretrained and fine-tuned models are licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt)).

The RGBX pretrained and fine-tuned models are licensed under ADOBE RESEARCH LICENSE(as defined in the [LICENSE-ADOBE](LICENSE-ADOBE.txt)).

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt), [LICENSE-MODEL](LICENSE-MODEL.txt), and [LICENSE-ADOBE](LICENSE-ADOBE.txt) respectively.
