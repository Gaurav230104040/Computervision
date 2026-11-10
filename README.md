# Deepfake Detection Using Frequency Space Learning with a Smooth Radial Frequency Mask

## 1. Project Overview

This project focuses on improving deepfake detection using frequency-domain feature learning. It is based on the original **FreqNet** architecture, which exploits frequency information to distinguish real images from AI-generated or manipulated images.

### Reference Paper

Tan et al., **"Frequency-Aware Deepfake Detection: Improving Generalizability through Frequency Space Learning" (AAAI 2024).**

- Original implementation: [FreqNet – GitHub](https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection)

### Proposed Novelty

The proposed modification introduces a **Smooth Radial Frequency Mask** in place of the rectangular hard frequency mask used in the original FreqNet implementation.

The original implementation uses a rectangular frequency selection mechanism. In the proposed approach, a radial mask is constructed using the distance of each frequency component from the center of the shifted Fourier spectrum.

The resulting mask provides a smooth radial transition rather than an abrupt rectangular frequency boundary.

### Key Modifications

1. Replaced the rectangular hard frequency mask with a smooth radial mask.
2. Used the distance from the center of the FFT-shifted spectrum to determine frequency selection.
3. Introduced a sigmoid-based transition to avoid abrupt frequency selection boundaries.
4. Retained the original FreqNet architecture and its remaining components.
5. Did not introduce additional learnable modules or modify the original loss function.

The proposed modification focuses specifically on improving frequency representation while preserving the underlying FreqNet architecture.



## 2. Environment and Dependencies

The implementation uses Python and PyTorch with CUDA acceleration.

### Requirements

Component 
 Python 
 PyTorch 
 CUDA 
 NumPy 
 OpenCV 
 TensorboardX

### Hardware
Component  
 GPU T4
 CPU 
 Operating System  Linux / Google Colab 

### Installation

Clone the original implementation:

bash
git clone https://github.com/chuangchuangtan/FreqNet-DeepfakeDetection.git

cd FreqNet-DeepfakeDetection


Install the dependencies:

bash
pip install -r requirements.txt


Install PyTorch with a CUDA-compatible build if it is not already installed.

Verify the installation:

python
import torch

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))




## 3. Dataset Preparation

The project uses the deepfake detection dataset specified in the experimental configuration.

**Dataset Download Link:** [https://huggingface.co/datasets/sywang/CNNDetection/resolve/main/progan_testset.zip?utm_source=chatgpt.com]

### Expected Directory Structure

The dataset should be organized according to the original implementation's expected data-loading configuration.

An example directory structure is:

text
dataset/
│
├── train/
│   ├── real/
│   └── fake/
│
├── validation/
│   ├── real/
│   └── fake/
│
└── test/
    ├── real/
    └── fake/

### Preprocessing

The preprocessing pipeline follows the original FreqNet implementation.

The general processing steps include:

1. Loading real and fake face images.
2. Applying the required image resizing and normalization.
3. Converting images into tensors.
4. Preparing training and evaluation batches.
5. Assigning binary class labels.

The binary classification setup uses:

- Real images: Class 0
- Fake images: Class 1

No additional learnable preprocessing module is introduced by the proposed radial frequency mask.



## 4. Training Instructions

Two implementations are considered:

- **Baseline:** Original FreqNet implementation.
- **Proposed:** Modified FreqNet implementation with a smooth radial frequency mask.

### 4.1 Baseline Training

The baseline uses the original FreqNet frequency filtering mechanism.

bash
python base_model.py
python trainer.py


The training configuration should specify:

- Dataset paths
- Number of classes
- Batch size
- Learning rate
- Number of epochs
- Optimizer
- Checkpoint output directory
- Training and validation configurations

### 4.2 Proposed Model Training

The proposed implementation modifies the frequency filtering operation in `hfreqWH()`.

The modified model is implemented in:

text
freqnet.py


The original implementation is retained separately in:

text
freqnet1.py

### 4.3 Training Configuration

The following parameters should match the experimental setup:

| Hyperparameter | Value |
|---|---|
| Model | FreqNet |
| Number of Classes | 1 (binary output) |
| Optimizer | Original FreqNet configuration |
| Learning Rate | Experimental configuration |
| Batch Size | Experimental configuration |
| Epochs | Experimental configuration |
| Loss Function | Original FreqNet loss |
| Frequency Mask | Smooth Radial Mask (Proposed) |
| Mask Scaling Factor | 4 |
| Sigmoid Transition Parameter | 0.1 |

The proposed approach does not introduce additional trainable parameters through the radial mask.

---

## 5. Inference and Evaluation

The trained checkpoints can be evaluated using the testing and validation scripts provided by the implementation.

### 5.1 Loading a Trained Checkpoint

The proposed checkpoint can be loaded using:

```python
import torch

checkpoint_path = "/content/4-classes-freqnet-v2.pth"

state_dict = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(state_dict, strict=True)
```

The model should be initialized using the appropriate implementation before loading the checkpoint.

### 5.2 Evaluation

The evaluation pipeline uses the original validation utilities and testing configuration.

The relevant files include:

```text
validate.py
test_options.py
util.py
```

The evaluation script should load the selected checkpoint, execute inference on the test dataset, and calculate the required evaluation metrics.

Example execution:

```bash
python test.py
```

The checkpoint path and dataset paths must be configured in the testing configuration.

### 5.3 Evaluation Metrics

The following metrics are used to evaluate the model:

- Accuracy (AC)
- Average Precision (AP)
  

The same dataset split and evaluation procedure should be used for both baseline and proposed implementations to ensure a fair comparison.

---

## 6. Checkpoints and Results

### 6.1 Model Checkpoints

The trained model checkpoints should be made accessible through the project repository or the provided submission link.

| Model | Checkpoint |
|---|---|
| FreqNet | https://drive.google.com/file/d/1vdZwsHMztj7IVl4_TVJ-RkpUSHf416jO/view?usp=sharing |


### 6.2 Experimental Results

The proposed model achieved the following reported results:

| Metric | Proposed FreqNet |
|---|---:|
| Average Precision (AP) | 100% |
| Accuracy (AC) | 99.7% |

The results indicate strong classification performance on the evaluated dataset.

### 6.3 Baseline Comparison

| Model | Average Precision (AP) | Accuracy (AC) |
|---|---:|---:|
| Original FreqNet | **100%** | **99.6%** |
| Proposed FreqNet | **100%** | **99.7%** |

The comparison is intended to evaluate the effectiveness of replacing the original rectangular frequency mask with a smooth radial frequency mask.

### 6.4 Additional Files

The following files should accompany the implementation:

- Original model checkpoint.
- Modified model checkpoint.
- Training configuration.
- Testing configuration.
- Training and validation logs.
- Evaluation results.

---

## 7. Code Organization and Reproducibility

### 7.1 Project Structure

The main implementation files are organized as follows:

```text
Deepfake-Detection/
│
├── freqnet1.py
│   └── Original FreqNet implementation
│
├── freqnet.py
│   └── Modified FreqNet implementation
│       with smooth radial frequency masking
│
├── validate.py
│   └── Validation and evaluation functions
│
├── test_options.py
│   └── Testing configuration
│
├── util.py
│   └── Utility functions
│
├── trainer.py
│   └── Model training entry point
│
├── test.py
│   └── Model testing and evaluation entry point
│
├── requirements.txt
│   └── Python dependencies
│
├── checkpoints/
│   ├── baseline/
│   └── proposed/
│
└── README.md
```

### 7.2 Identifying the Proposed Modification

The main proposed modification is located in the `hfreqWH()` function of `freqnet.py`.

The original implementation uses rectangular frequency masking, whereas the modified implementation constructs a radial distance map and applies a sigmoid-based smooth mask.

The rest of the architecture is retained to ensure that the experimental comparison primarily evaluates the effect of the proposed frequency filtering modification.

### 7.3 Reproducibility Procedure

To reproduce the experiments:

1. Clone the repository.
2. Install the required dependencies.
3. Download and prepare the dataset.
4. Configure the dataset and checkpoint paths.
5. Train the original FreqNet baseline.
6. Train the proposed radial-mask FreqNet.
7. Load the corresponding trained checkpoints.
8. Run evaluation on the same test split.
9. Compare the resulting AP and accuracy metrics.

Both models should be evaluated under identical experimental conditions.

---

## 8. Conclusion

This project investigates the effect of smooth radial frequency masking on frequency-space learning for deepfake detection.

By replacing the rectangular hard frequency mask in the original FreqNet implementation with a sigmoid-based radial mask, the proposed approach introduces a smoother frequency selection mechanism while retaining the original network architecture.

The proposed implementation achieved **100% Average Precision and 99.7% Accuracy** in the reported evaluation.

The implementation and experimental setup are intended to provide a reproducible comparison between the original FreqNet architecture and the proposed frequency masking modification.
