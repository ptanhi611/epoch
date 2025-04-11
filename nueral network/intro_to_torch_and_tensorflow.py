import torch
import os
import numpy as np


os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# print(tf.__version__)
# print(torch.__version__)

tensor1=torch.randn(3,3,3,4)
print(tensor1)