"""
infer_handwritten.py
Load a saved CNN model and predict the class of a single image (28x28 or resized to 28x28).
Usage:
    python src/infer_handwritten.py --model ../models/handwritten_cnn.h5 --image path/to/image.png
"""
import argparse
from pathlib import Path
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

def preprocess_image(path):
    img = Image.open(path).convert('L').resize((28,28))
    arr = np.array(img).astype('float32') / 255.0
    arr = arr.reshape(1,28,28,1)
    return arr

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--image', required=True)
    args = parser.parse_args()

    x = preprocess_image(args.image)
    model = load_model(args.model)
    p = model.predict(x)
    label = int(p.argmax(axis=1)[0])
    prob = float(p.max())
    print('Predicted label:', label, 'probability:', prob)

if __name__ == '__main__':
    main()
