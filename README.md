# CTDM-Net

This repository provides the official implementation of **CTDM-Net: A CNN-Transformer Dynamic Memory Network for Cropland Semantic Change Detection**.

## Pre-trained Weights

The trained model weights are available at **
 https://pan.baidu.com/s/1Dq-OfG9TehC5umTlJW4aHw?pwd=ej9r**.

Please download the weights and place them in the corresponding path specified in the configuration file.

## Environment

The code is implemented with **Python 3.9**.

Install the required dependencies with:

```bash
pip install -r requirements.txt
```

## Dataset Preparation

CTDM-Net currently provides configuration files for the following datasets:

- CropSCD
- JL1
- FZSCD
- SECOND

Please prepare the corresponding datasets and update the dataset paths in the configuration files under `config_fusion/`.

## Training

Training is performed through the configuration files in `config_fusion/`.

For example:

```bash
python train.py --config config_fusion/CropSCD.yaml
```

Other datasets can be trained by specifying the corresponding configuration file:

```bash
python train.py --config config_fusion/JL1.yaml
```

## Prediction

To generate prediction results using a trained model:

```bash
python predict.py --config config_fusion/JL1.yaml --model_dir path/to/model.pth --save_dir results/JL1
```

The predicted semantic change maps will be saved to the specified output directory.

## Evaluation

To evaluate the generated prediction maps:

```bash
python eval.py --test_file path/to/test.txt --prediction_dir path/to/predictions --label_dir path/to/labels --num_classes 9
```

## Quick Start

```bash
git clone https://github.com/ZhuOO5/CTDM-Net.git
cd CTDM-Net
pip install -r requirements.txt
```

Then prepare the dataset, update the corresponding configuration file, and run the training or prediction script.

## Project Structure

```text
CTDM-Net/
├── config_fusion/      # Dataset and experiment configurations
├── loss/               # Loss functions
├── models/             # CTDM-Net model implementation
├── results/            # Prediction results
├── data_utils.py       # Dataset utilities
├── metrics.py          # Evaluation metrics
├── train.py            # Training
├── predict.py          # Prediction
├── eval.py             # Evaluation
├── run.sh              # Training examples
└── requirements.txt    # Python dependencies
```

