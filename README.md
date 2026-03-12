CapsNet-PyTorch
A streamlined implementation of Capsule Networks (CapsNets) using PyTorch.

Traditional Convolutional Neural Networks (CNNs) often struggle with spatial orientation—they might recognize a face even if the eyes and mouth are swapped. CapsNets solve this by using Dynamic Routing to preserve the hierarchical relationships between features.

🚀 Key Features
Spatial Awareness: Better at identifying objects regardless of rotation or scale.

Dynamic Routing: Implements the routing-by-agreement algorithm (Sabour et al., 2017).

Modular Design: Easy-to-read code designed for experimentation on MNIST and similar datasets.

🛠️ Quick Start
1. Installation
Clone the repo and install the requirements:

Bash
git clone https://github.com/yakshaxo/capsnet.git
cd capsnet
pip install -r requirements.txt
2. Training
Start training with default parameters:

Bash
python train.py
3. Evaluation
Test your trained model's accuracy:

Bash
python evaluate.py --model_path ./path_to_model.pth
📈 Performance
CapsNet is designed to achieve high accuracy on MNIST (~99%) while requiring significantly less data to understand spatial variances compared to standard CNNs.



Classification of Images using Capsule Networks (CapsNet)

Traditional Convolutional Neural Networks (CNNs) rely on "Pooling" layers, which often discard critical information regarding the position and orientation of objects. This project implements a Capsule Network architecture that treats features as vectors rather than scalars.

How it works:

Vector Representations: Instead of simple neurons, the model uses "Capsules"—groups of neurons that represent the properties (pose, scale, texture) of an entity.

Dynamic Routing: Replaces Max-Pooling with a "Routing-by-Agreement" mechanism. This ensures that the network understands that a face is only a face if the eyes, nose, and mouth are in the correct relative positions.

Spatial Invariance: Achieves significantly better performance on rotated or skewed datasets compared to traditional deep learning models.

Maintained by krayem louay


