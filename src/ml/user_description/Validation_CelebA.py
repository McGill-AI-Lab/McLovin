import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

#defining the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
lr = 0.0001
max_epoch = 4
batch_size = 32

#Defining which columns I wish to preserver
RELEVANT_ATTRIBUTES = [0, 4, 5, 6, 7, 8, 9, 11, 12, 13, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 29, 30, 31, 32, 38]

class CustomCelebADataset(Dataset):
    def __init__(self, dataset, relevant_attributes, transform=None):
        self.dataset = dataset  # Original CelebA dataset
        self.relevant_attributes = relevant_attributes  # List of indices of the relevant attributes
        self.transform = transform  # Image transformation (if any)

    def __getitem__(self, idx):
        image, labels = self.dataset[idx]  # Get image and labels
        
        # Keep only the relevant attributes
        labels = labels[self.relevant_attributes] #Tensor indexing is possible since we will be working with torch.Tensor()
        
        # Apply transformations if defined
        if self.transform:
            image = self.transform(image)
        
        return image, labels

    def __len__(self):
        return len(self.dataset)  # Return the total number of samples in the dataset

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

#Defining the datasets for testing
test_dataset = torchvision.datasets.CelebA(root='./data', split='test', target_type='attr',
                                             download=True, transform=transform)

#Creating the custom dataset for testing (ie removing irrelevant columns)
test_dataset = CustomCelebADataset(test_dataset, RELEVANT_ATTRIBUTES, transform=None)

#Loading the testing data
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

#Defining the model
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        #defining the convolutional layers as well as the pool layer
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)

        #this is the final fully connected layer
        self.fc1 = nn.Linear(128 * 30 * 30, 120)  # fully connected layer #(W-F + 2P)/S + 1
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):

        #the order of computation is conv, then relu, then pool. Since there
        #are two layers, this is done twice
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 30 * 30)  # FLATTENS THE TENSOR
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

#defining the model
FILE = "model.pth"
loaded_model = ConvNet().to(device)
loaded_model.load_state_dict(torch.load(FILE, map_location=device))
loaded_model.eval()

print('Starting the validation loop')
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    n_class_correct = [0 for i in range(26)]
    n_class_samples = [0 for i in range(26)]

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device).float() #target must be float type
        outputs = loaded_model(images)

        probabilities = torch.sigmoid(outputs)

        #convert probabilities into a binary assessement with 0.5 being the threshold, 
        predicted = (probabilities > 0.5).float()

        n_samples += labels.size(0) #this should represent the batch size

        # For each attribute, update correct and total counts
        for j in range(26):
            n_class_correct[j] += ((predicted[:, j] == labels[:, j])).sum().item()
            n_class_samples[j] += labels[:, j].size(0)
        
        if (i+1)%10 == 0:
            print(i, 'batches have been completed')

    for i in range(26):
        accuracy = 100.0 * n_class_correct[i] / n_class_samples[i] if n_class_samples[i] > 0 else 0.0
        print(f"Attribute {i}: Accuracy = {accuracy:.2f}%")


