import sys
import os
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

try:
    from MLP.dense import Layer_Dense
    print("Layer_Dense import successful")
except ImportError as e:
    print(f"Error importing Layer_Dense: {e}")

try:
    from MLP.activation_function import Sigmoid
    print("Sigmoid import successful")
except ImportError as e:
    print(f"Error importing Sigmoid: {e}")
