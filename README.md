# Sudoku-RWKV

A specialized RWKV model for solving Sudoku puzzles.

![menu](./assets/menu.png)

## Requirements

- rwkv
- tkinter

## Quick Start

- Run `demo.py` or `minimum_inference.py` to solve Sudoku puzzles
- Run `generate_sudoku_data.py` to generate training data

## Model

The current `sudoku_rwkv_241116.pth` model is a specialized RWKV-v6 model trained on 750k Sudoku samples (~14.4B tokens) specifically for solving Sudoku puzzles.

Model specifications:
- Parameters: ~12.7M
- Vocabulary size: 133
- Architecture: 8 layers, 256 dimensions

The model includes a simple improvement for better performance (see `model.py` line 372). Corresponding modifications were made in the inference code (`rwkv_model.py` lines 852, 893-896).

## Training

The model was trained using the [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) repository.

Hyperparameters:
- `M_BSZ`: 48
- `CTX_LEN`: 8192
- `LR`: 8e-4 to 3e-5
- `ADAM_EPS`: 1e-18
- `ADAM_BETA1`: 0.9
- `ADAM_BETA2`: 0.95
- `WEIGHT_DECAY`: 0.1

Loss Curve:
![Training Loss Curve](./assets/loss.png)

## Experiments

- Below are the old results. The current model seems to be able to solve any solvable Sudoku. If you find any failed cases, please let me know.

I tested the model on samples of varying difficulty levels, with results shown below:

Note: Difficulty is measured by the number of empty cells in the Sudoku puzzle

![Accuracy Results](./assets/perfect_solution_rate.png)

![Token Usage](./assets/token_usage.png)
