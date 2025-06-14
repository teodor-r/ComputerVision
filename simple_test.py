import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

import torchvision
print(f"Torchvision version: {torchvision.__version__}")

import matplotlib
print(f"Matplotlib version: {matplotlib.__version__}")

import sklearn
print(f"Scikit-learn version: {sklearn.__version__}")