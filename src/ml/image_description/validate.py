import torch
import torch.nn as nn
import torch.nn.functional as F

from src.ml.image_description.config import DEVICE, BATCH_SIZE, MODEL_PATH
from src.ml.image_description.data.celeba_dataset import create_celeba_dataloader
from src.ml.image_description.models.conv_net import ConvNet

def validate_celeba():
    # Create the DataLoader for test split
    test_loader = create_celeba_dataloader(split='test', batch_size=BATCH_SIZE, shuffle=False)

    # Load model
    model = ConvNet(num_classes=26).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print('Starting the validation loop...')
    n_class_correct = [0] * 26
    n_class_samples = [0] * 26

    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader):
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images)
            probabilities = torch.sigmoid(outputs)
            predicted = (probabilities > 0.5).float()

            for j in range(26):
                n_class_correct[j] += (predicted[:, j] == labels[:, j]).sum().item()
                n_class_samples[j] += labels[:, j].size(0)

            if (i + 1) % 10 == 0:
                print(f'{i+1} test batches completed')

    # Print accuracy for each attribute
    for idx in range(26):
        accuracy = 0.0
        if n_class_samples[idx] > 0:
            accuracy = 100.0 * n_class_correct[idx] / n_class_samples[idx]
        print(f"Attribute {idx}: Accuracy = {accuracy:.2f}%")

if __name__ == "__main__":
    validate_celeba()
