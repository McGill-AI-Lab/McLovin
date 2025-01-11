import time
import torch
import torch.nn as nn
import torch.optim as optim

from src.ml.image_description.config import DEVICE, LR, MAX_EPOCH, BATCH_SIZE, MODEL_PATH
from src.ml.image_description.data.celeba_dataset import create_celeba_dataloader
from src.ml.image_description.models.conv_net import ConvNet

def train_celeba():
    # Create the DataLoader for train split
    train_loader = create_celeba_dataloader(split='train', batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate the model
    model = ConvNet(num_classes=26).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    n_steps = len(train_loader)

    for epoch in range(MAX_EPOCH):
        start_time = time.time()

        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{MAX_EPOCH}], Step [{i+1}/{n_steps}], Loss: {loss.item():.4f}')

        elapsed_time = time.time() - start_time
        print(f'Time for epoch {epoch+1}: {elapsed_time:.2f} seconds')

    print('Finished Training! Saving model...')
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model saved to {MODEL_PATH}')

if __name__ == "__main__":
    train_celeba()
