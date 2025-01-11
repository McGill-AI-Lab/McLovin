# here are the parameters and all configurations used for the image_description
import torch

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
LR = 0.0001
MAX_EPOCH = 4
BATCH_SIZE = 32

# Indices for the columns in CelebA we want to keep
RELEVANT_ATTRIBUTES = [
    0, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18,
    19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 38
]

# Model save location
MODEL_PATH = "src/ml/image_description/models/model.pth"
