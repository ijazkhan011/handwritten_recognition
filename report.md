# Handwritten Character Recognition — Report

Objective:
Recognize handwritten characters using CNNs. This project uses MNIST for demonstration and can be extended to EMNIST for characters.

Architecture:
- Two Conv2D + MaxPooling blocks, Flatten, Dense(128), Dropout, Dense(num_classes).

Evaluation:
- Use accuracy, confusion matrix, and per-class precision/recall/F1. For EMNIST, consider data augmentation (rotations, shifts) and class balancing.

Next steps:
- Train on EMNIST, add augmentation, and convert to CRNN for word-level recognition.
