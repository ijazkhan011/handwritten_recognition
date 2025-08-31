"""
train_handwritten.py
Train a CNN on MNIST (digits) or a preprocessed EMNIST dataset.
Usage:
    python src/train_handwritten.py --dataset mnist --out_model ../models/handwritten_cnn.h5 --epochs 10
For EMNIST: preprocess EMNIST into numpy arrays (x_train,x_test,y_train,y_test) and adjust the script to load them.
"""
import argparse
from pathlib import Path
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', choices=['mnist','emnist'], default='mnist')
    parser.add_argument('--out_model', required=True)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.astype('float32') / 255.0
        x_test = x_test.astype('float32') / 255.0
        x_train = x_train[..., None]
        x_test = x_test[..., None]
        num_classes = 10
    else:
        # Expect preprocessed npz at data/emnist.npz with x_train,x_test,y_train,y_test
        data_file = Path('data/emnist.npz')
        if not data_file.exists():
            raise FileNotFoundError('EMNIST file not found at data/emnist.npz. Preprocess EMNIST into that file.')
        data = np.load(data_file)
        x_train = data['x_train'].astype('float32') / 255.0
        x_test = data['x_test'].astype('float32') / 255.0
        x_train = x_train[..., None]
        x_test = x_test[..., None]
        y_train = data['y_train']
        y_test = data['y_test']
        num_classes = int(np.max(y_train)) + 1

    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    model = build_cnn(x_train.shape[1:], num_classes)
    es = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    model.fit(x_train, y_train_cat, epochs=args.epochs, batch_size=128, validation_split=0.1, callbacks=[es])
    model.save(args.out_model)
    print('Saved model to', args.out_model)

if __name__ == '__main__':
    main()
