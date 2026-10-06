"""
test_imports.py
Validates that all critical dependencies are installed and functional.
"""

import sys

def test_imports():
    print("Testing core imports...")
    try:
        import torch
        print(f"  PyTorch: {torch.__version__} (CUDA: {torch.cuda.is_available()})")
        import torchvision
        print(f"  TorchVision: {torchvision.__version__}")
        import cv2
        print(f"  OpenCV: {cv2.__version__}")
        import numpy as np
        print(f"  NumPy: {np.__version__}")
        import pandas as pd
        print(f"  Pandas: {pd.__version__}")
        import sklearn
        print(f"  Scikit-Learn: {sklearn.__version__}")
        import scipy
        print(f"  SciPy: {scipy.__version__}")
        import matplotlib
        print(f"  Matplotlib: {matplotlib.__version__}")
        import streamlit
        print(f"  Streamlit: {streamlit.__version__}")
        import datasets
        print(f"  HuggingFace Datasets: {datasets.__version__}")
        import statsmodels
        print(f"  Statsmodels: {statsmodels.__version__}")
        print("\nAll dependencies imported successfully!")
    except ImportError as e:
        print(f"Import error: {e}")
        raise

if __name__ == "__main__":
    test_imports()
    sys.exit(0)
