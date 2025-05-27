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
├── auxiliary_experiments/    # Extra scripts and analyses (face ratio, bias metrics, etc.)
├── backbones/                 # Backbone model definitions (e.g., ResNet, Swin-Transformer)
├── checkpoint/                # Saved model checkpoints
├── class_label.json           # Class mapping for evaluation
├── dataset/                   # Main image dataset (train/val/test splits)
├── dataset_generation/        # Scripts for dataset crawling and construction
├── eval.py                    # Model evaluation and Grad-CAM visualization script
├── README.md                  # Project documentation (this file)
├── run.sh                     # Quick launch script for training/evaluation
├── train.py                   # Main training script (supports multiple backbones)
└── util/                      # Utility functions (GradCAM, data helpers, etc.)
```

## Usage

### 1. Environment Setup

Install required packages:
```bash
pip install torch torchvision timm numpy tqdm pillow opencv-python 
```

### 2. Dataset Preparation

Our dataset is a hybrid of real-world crawled images and high-quality synthetic data generated via SDXL. Follow these steps to reproduce the dataset:

**Recommended:**  
To get started quickly, simply run:

```bash
chmod +x ./util/download_dataset.sh 
./util/download_dataset.sh
```

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
chmod +x make_dataset.sh
python make_dataset.sh
```

This script merges crawled and synthetic data (paths are configurable within the script).
The merged dataset will be placed in `../dataset/`.

**Gender Proportion Splits:**

The dataset will be automatically split into five groups with different female ratios: 0%, 50%, 100%. Each group can be used for downstream bias and fairness experiments.

**Directory Structure**

After completion, your `./dataset/` directory will look like:

```
dataset/
├── dataset_0/
├── dataset_50/
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
    chmod +x ./util/download_pretrain_weight.sh
    ./util/download_pretrain_weight.sh
    ```

- (Recommended) To download the fully-trained model checkpoint for evaluation or comparison:

    ```bash
    chmod +x ./util/download_baseline.sh 
    ./util/download_baseline.sh 
    ```

Both scripts will automatically place the weights into the `./checkpoint/` directory.

**Training From Scratch**
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

To apply dlib-based mask generation to your dataset, use the provided script:

1. **Navigate** to the `dataset_generation/Mask_add_Generator` directory:
```bash
    cd dataset_generation/Mask_add_Generator
```

2. **Run the script** with your input folder. For example:
```bash
    python add_mask_dlib.py --input_root ../../dataset/dataset_0
```
    This will generate a face-masked dataset in `../../dataset/dataset_0_mask`.

3. **Modify** the training dataset path in your configuration or script to point to the newly generated masked dataset, then run `train.py` as usual.

> **Tip:**  
> You can specify a custom output directory using the `--output_root` argument if desired.  
> Example:
> ```bash
> python add_mask_dlib.py --input_root ../../dataset/dataset_0 --output_root ../../dataset/dataset_0_masked
> ```

Make sure your input and output paths are correct and have the necessary read/write permissions.

### 6. Running All Experiments
Training metrics will be automatically recorded in the checkpoint files.
There are two auxiliary experiments available in `./auxiliary_experiments`:

**1: Confusion Matrix**
```bash
chmod +x plot_cf.sh
./plot_cf.sh
```
The shell script will automatically plot all confusion matrices using the pretrained weights located in `./checkpoint`.

**2:Face Ratio**
```bash
python script.py \
    --data_root ../dataset/dataset_100/val \
    --model_path ../checkpoint/swin_dataset_100_Mask_20250524_141628/best_model.pth \
    --model_name swin \
    --num_classes 10 \
    --threshold 0.2
```
This script computes the Face Ratio, which quantifies what percentage of the GradCAM attribution falls within the detected face region of each image.

## References

1. R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks,” ICCV, 2017.
2. A. Chattopadhay et al., “Grad-CAM++: Improved Visual Explanations for Deep Convolutional Networks,” WACV, 2018.

## Contributors

- Xiao Gao (xgao045)
- Maojie Tang (mtang096)  
_Group 5, CS229 Project_
