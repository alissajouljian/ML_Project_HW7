# Machine Learning Pipeline

The Machine Learning Pipeline is a tool designed to preprocess and model data for training and testing purposes. It provides functionality for both training models on input data and using pre-trained models for making predictions. This README provides an overview of the project, installation instructions, usage guidelines, and other relevant details.

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Arguments](#arguments)
5. [Example](#example)

## Introduction

The Machine Learning Pipeline comprises several modules, including a model module (`model.py`), a preprocessor module (`preprocessor.py`), and the main script (`main.py`). These modules work together to preprocess input data, train machine learning models, and generate predictions.

## Installation

To install the Machine Learning Pipeline and its dependencies, follow these steps:

1. Clone the repository:

```bash
git clone <repository_url>
```

2. Navigate to the project directory:

```bash
cd machine-learning-pipeline
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

The Machine Learning Pipeline can be used for both training models and making predictions. The main script (`main.py`) accepts command-line arguments to specify the data file path and whether to run in test mode.

To train a model, use the following command:

```bash
python main.py --data_path <path_to_data_file>
```

To run in test mode and generate predictions using a pre-trained model, use:

```bash
python main.py --data_path <path_to_data_file> --inference True
```

## Arguments

The main script accepts the following command-line arguments:

- `--data_path`: Path to the input data file (required).
- `--inference`: Optional argument to activate test mode (default: False).
- `--algorithm`: Optional argument to specify the training model (default: "GBoost").

## Example

Here's an example of how to use the Machine Learning Pipeline:

```bash
python main.py --data_path data.csv --inference True
```

This command will load a pre-trained model, preprocess the data from `data.csv`, and generate predictions.
