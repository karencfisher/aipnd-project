# AI Programming with Python Project

This repository contains Karen Fisher's project code for Udacity's **AI Programming with Python** Nanodegree program. The project involves building an image classifier using PyTorch and converting it into a command-line application.

## Project Overview

The goal of this project is to develop an image classifier capable of identifying different species of flowers. The classifier is trained on a dataset of flower images and can predict the species of a flower from an input image. The project is divided into two main components:

1. **Training the Model**: Using a dataset of flower images, a neural network is trained to classify images into different categories.
2. **Command-Line Applications**: 
- *train.py*: Trains a classifier using transfer learning from a given Torchvision pretrained model.
- *predict.py*: Runs inference on a single image using a previous trained classifier.

## Repository Structure

```
├── CLIApp/
│   ├── __init__.py
│   ├── predict.py                  # Script for making predictions
│   ├── train.py                    # Script for training the model
│   ├── utilities.py                # Helper functions
│   ├── arg-templates/              # JSON templates for CLI arguments
├── flowers/                        # Dataset of flower images
│   ├── train/                      # Training images
│   ├── valid/                      # Validation images
│   ├── test/                       # Test images
├── models/                         # Saved model checkpoints
├── assets/                         # Additional resources
│   ├── cat_to_name.json            # Mapping category IDs to names
├── Image_Classifier_Project.ipynb  # Notebook training/testing a flower classifier
├── README.md                       # Project documentation
├── environment.yml                 # Conda environment configuration
├── pyproject.toml                  # Project dependencies
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/aipnd-project.git
   cd aipnd-project
   ```

2. **Set Up the Environment**:
   You can set up the environment using either Conda or Poetry.

   **Option 1: Conda**
   Create and activate the Conda environment using the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   conda activate aipnd-project
   ```

   **Option 2: Poetry**
   Install Poetry if you don't already have it:
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```
   Then install the dependencies and activate the environment:
   ```bash
   poetry install
   poetry shell
   ```

## Usage

### Training the Model
To train the model, use the `train.py` script. Example:
```bash
python -m CLIApp.train --data_dir flowers --save_dir models --arch vgg16 --learning_rate 0.001 --hidden_units 512 --epochs 10 --gpu
```

### Making Predictions
To make predictions on new images, use the `predict.py` script. Example:
```bash
python -m CLIApp.predict --image_file_path path/to/image.jpg --model_checkpoint models/checkpoint.pth --category_names assets/cat_to_name.json --top_k 5 --gpu
```

## Features
- Train a neural network on a dataset of flower images.
- Save and load model checkpoints.
- Predict the species of a flower from an input image.
- Command-line interface for training and prediction.

## Acknowledgments
- This project is part of Udacity's AI Programming with Python Nanodegree program.
- The flower dataset is provided by Udacity.
- PyTorch is used for building and training the neural network.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Credits
This README was drafted with the assistance of GitHub Copilot.
