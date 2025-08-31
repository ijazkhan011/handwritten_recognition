# Handwritten Character Recognition (Task 3)

## Objective
Identify handwritten characters (digits or alphabets) using image processing and deep learning (CNN).

## Datasets
- MNIST (digits) — used by default for demo.
- EMNIST (characters) — preprocess and save as `data/emnist.npz` with `x_train,x_test,y_train,y_test` arrays.

## How to run
1. Install requirements:
```
pip install -r requirements.txt
```
2. Train on MNIST (example):
```
python src/train_handwritten.py --dataset mnist --out_model models/handwritten_cnn.h5 --epochs 10
```
3. Predict a single image:
```
python src/infer_handwritten.py --model models/handwritten_cnn.h5 --image path/to/image.png
```
