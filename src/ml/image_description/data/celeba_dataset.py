import torch
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
from src.ml.image_description.config import RELEVANT_ATTRIBUTES  # or a relative import if needed

class CustomCelebADataset(Dataset):
    def __init__(self, dataset, relevant_attributes, transform=None):
        self.dataset = dataset
        self.relevant_attributes = relevant_attributes
        self.transform = transform

    def __getitem__(self, idx):
        image, labels = self.dataset[idx]
        labels = labels[self.relevant_attributes]  # keep only relevant attributes

        if self.transform:
            image = self.transform(image)

        return image, labels

    def __len__(self):
        return len(self.dataset)
    
def center_images(image):
    centered_image = image
    return centered_image


def get_celeba_transforms():
    """Return the Compose of transforms used for CelebA images."""
    return transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

def create_celeba_dataloader(split: str, root: str = './src/ml/image_description/data', download: bool = True,
                             batch_size: int = 32, shuffle: bool = True):
    """
    Utility function to create a DataLoader for CelebA (train, val, test).
    """
    transform = get_celeba_transforms()
    celeba_dataset = torchvision.datasets.CelebA(
        root=root, split=split, target_type='attr', download=download, transform=transform
    )

    # Wrap dataset in our custom dataset to keep only the relevant attributes
    custom_dataset = CustomCelebADataset(celeba_dataset, RELEVANT_ATTRIBUTES, transform=None)
    dataloader = torch.utils.data.DataLoader(
        custom_dataset, batch_size=batch_size, shuffle=shuffle
    )
    return dataloader
