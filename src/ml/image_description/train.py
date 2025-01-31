import time
import torch
import torch.nn as nn
import torch.nn.utils
import torch.optim as optim

from src.ml.image_description.config import DEVICE, LR, MAX_EPOCH, BATCH_SIZE, MODEL_PATH
from src.ml.image_description.data.celeba_dataset import create_celeba_dataloader
from src.ml.image_description.models.conv_net import ConvNet

def train_celeba():

    train_loader = create_celeba_dataloader(split='train', batch_size=BATCH_SIZE, shuffle=True)
    val_loader = create_celeba_dataloader(split='valid', batch_size=BATCH_SIZE)  # Add validation

    model = ConvNet(num_classes=26).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)

    best_val_loss = float('inf')
    patience_counter = 0
    max_patience = 5  # Early stopping
    
    for epoch in range(MAX_EPOCH):
        # Training
        
        model.train()
        train_loss = 0
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(DEVICE)
            labels = labels.float().to(DEVICE)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f'Epoch [{epoch+1}/{MAX_EPOCH}], Step [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f}')

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)
                labels = labels.float().to(DEVICE)
                outputs = model(images)
                val_loss += criterion(outputs, labels).item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        print(f'Epoch [{epoch+1}/{MAX_EPOCH}], '
              f'Train Loss: {train_loss/len(train_loader):.4f}, '
              f'Val Loss: {val_loss:.4f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), MODEL_PATH)
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= max_patience:
            print("Early stopping triggered")
            break


if __name__ == "__main__":
    train_celeba()
