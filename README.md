# Interpreting Gender Bias in Vision Models via Grad-CAM Variants

## Project Overview

This project investigates how state-of-the-art vision models may inherit and amplify gender bias—such as associating “doctor” with men and “nurse” with women—when classifying professions. We aim to answer:  
- **Do vision models rely on gender-related visual features for profession recognition?**
- **How does this reliance differ between CNNs (ResNet50) and Transformers (Swin-Transformer)?**
- **How do Grad-CAM and Grad-CAM++ highlight or explain these biases?**

Our approach uses both Grad-CAM and Grad-CAM++ to visualize and ablate gender-activated regions, quantifying the impact of gender on profession classification. We conduct experiments on a custom hybrid dataset, which consists of web-crawled real-world images and SDXL-generated synthetic images, all filtered to include explicit gender and profession labels.

## Directory Structure

```
.
├── backbones/            # Backbone model definitions (ResNet, Swin-Transformer)
├── checkpoint/           # Saved model checkpoints
├── class_label.json      # Class-to-label mapping for consistent evaluation
├── dataset/              # Image dataset (with train/val splits)
├── dataset_generation/   # Scripts for filtering/constructing custom datasets
├── eval.py               # Evaluation and Grad-CAM visualization
├── log/                  # Training logs
├── README.md             # Project documentation (this file)
├── run.sh                # Shell script for quick training/evaluation launch
├── train.py              # Main training script (multi-backbone, multi-head)
└── util/                 # Utilities (GradCAM, loss functions, etc.)
```


## Usage

### 1. Environment Setup

Install required packages:
```bash
pip install torch torchvision timm numpy tqdm pillow opencv-python 
```

### 2. Dataset Preparation

Our dataset is a hybrid of real-world crawled images and high-quality synthetic data generated via SDXL. Follow these steps to reproduce the dataset:

#### **Step 1: Web Image Crawling**

Run the following script to automatically collect real-world images for each profession keyword.
The search keywords are specified in `./dataset_generation/crawler_generator.py`.

```bash
python ./dataset_generation/crawler_generator.py
```

Crawled images will be saved under `./dataset_generation/Crawler_data/`.

#### **Step 2: Synthetic Data Generation with SDXL**

For each profession, generate synthetic images with a controlled gender ratio (1:1) using SDXL and LLM-designed prompts:

**Prompt Generation:**

Generate prompts for SDXL using ChatGPT, ensuring balanced male/female representation.
Run the script to build prompts:

```bash
python ./dataset_generation/SDxl_Generator/build_prompt.py
```

By default, 500 prompts are created per profession (this is modifiable in the script).
All prompts are saved in sdxl_prompts.csv.

**Parallel Image Generation:**

Generate images using SDXL (see Stable Diffusion XL on HuggingFace).
Use the provided `run.sh` script to generate images in parallel on multiple GPUs.
The generated SDXL images will be stored in the configured output directory.

#### **Step 3: Dataset Assembly & Splitting**

After both Crawler and SDXL images are ready:

Navigate to the `./dataset_generation/` directory.
Run the script to merge data and create splits:

```bash
python make_dataset.py
```

This script merges crawled and synthetic data (paths are configurable within the script).
The merged dataset will be placed in `./dataset/`.

**Gender Proportion Splits:**

The dataset will be automatically split into five groups with different female ratios: 0%, 25%, 50%, 75%, 100%. Each group can be used for downstream bias and fairness experiments.

**Directory Structure**

After completion, your `./dataset/` directory will look like:

```
dataset/
├── dataset_0/
├── dataset_25/
├── dataset_50/
├── dataset_75/
└── dataset_100/
    ├── train/
    └── val/
```

Each subfolder (e.g., `dataset_0`) contains profession-labeled subfolders with images.

For more details or to customize data paths and proportions, refer to the script files and their inline comments.


### 3. Training
**Pretrained Weights**

Before training, make sure to download the necessary pretrained weights:

- To download backbone weights (for training from scratch or finetuning):

    ```bash
    ./util/download_pretrain_weight.sh
    ```

- To download the baseline model checkpoint for evaluation or comparison:

    ```bash
    ./util/download_baseline.sh
    ```

Both scripts will automatically place the weights into the `./checkpoint/` directory.

**Training From Scracth**
To train a model (e.g., ResNet50 with dual-head for gender and profession), run:
```
python train.py \
    --model resnet50 \
    --data_root ./dataset/dataset_100 \
    --pretrain_path ./checkpoint/resnet50-19c8e357.pth \
    --save_path ./checkpoint/resnet50_dataset_100.pth \
    --epochs 50 \
    --batch_size 128
```
Options:
- `--model`: Model backbone, choose from `resnet50` or `swin`
- `--use_focal_loss`: Add this flag to use focal loss in addition to cross entropy (default: off)

### 4. Evaluation and Visualization
To visualize Grad-CAM and Grad-CAM++ for gender/profession prediction, run:
```
python eval.py \
    --model resnet50 \
    --ckpt ./checkpoint/resnet50_dataset_100.pth \
    --img ./dataset/dataset_0/val/doctor/000001.jpg \
    --labels class_label.json \
    --use_gradcam \
    --use_gradcampp \
```
Results will include the original image and Grad-CAM/Grad-CAM++ heatmaps.

### 5. Masking-based Ablation

### 6. Running All Experiments

## Key Features

- **Dual-head classifier:** Joint gender/profession prediction.
- **Backbone comparison:** Switch between ResNet50 (CNN) and Swin-Transformer (Transformer).
- **Explainable AI:** Integrated Grad-CAM and Grad-CAM++.
- **Bias quantification:** Masking-based ablation to measure gender influence.
- **OpenImages compatible:** Dataset filtering and mapping scripts provided.

## References

1. R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks,” ICCV, 2017.
2. A. Chattopadhay et al., “Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks,” WACV, 2018.

## Contributors

- Xiao Gao (xgao045)
- Maojie Tang (mtang096)  
_Group 5, CS229 Project_
