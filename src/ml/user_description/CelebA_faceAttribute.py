import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

import matplotlib.pyplot as plt


#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameters
lr = 0.001
max_epoch = 8
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

#Defining the datasets for training and testing
train_dataset = torchvision.datasets.CelebA(root='./data', split='train', target_type='attr',
                                             download=True, transform=None)
test_dataset = torchvision.datasets.CelebA(root='./data', split='test', target_type='attr',
                                             download=True, transform=None)

#Creating the custom dataset for training and testing (ie removing irrelevant columns)
train_dataset = CustomCelebADataset(train_dataset, RELEVANT_ATTRIBUTES, transform=transform)
test_dataset = CustomCelebADataset(test_dataset, RELEVANT_ATTRIBUTES, transform=transform)

#Loading the training and testing data
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

'''
for i, (images, labels) in enumerate(train_loader):
    print(images.size()) #torch.Size([32, 3, 128, 128])
    print(labels.size()) #torch.Size([32, 26])
    print(images.dtype) #torch.float32
    print(labels.dtype) #torch.int64
    break
'''

#Defining the model
class ConvNet(nn.Module):
    def __init__(self):
        #calling the init of the super function
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

        #no relu since this will be applied in the loss function by ConvNet
        x = self.fc3(x)
        return x

#Creating an instance to our ConvNet model
model = ConvNet().to(device)

#Establishing our criterion and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr = lr)


n_steps = len(train_loader)

#Defining the training loop
for epoch in range(max_epoch):
    for i, (images, labels) in enumerate(train_loader):

        images = images.to(device)
        labels = labels.float().to(device) #ConvNet must take a float for labels

        #forward pass first
        outputs = model(images)
        loss = criterion(outputs, labels)

        #backward pass and optimizer
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print(f'epoch {epoch+1}/ {max_epoch}, step {i+1}/{n_steps}, loss = {loss.item():4f}')

print('Finished Training!')
FILE = "model.pth"
torch.save(model.state_dict(), FILE)

'''
print('Starting the validation loop')
with torch.no_grad():
    n_correct = 0
    n_samples = 0

    n_class_correct = [0 for i in range(26)]
    n_class_samples = [0 for i in range(26)]

    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        probabilities = torch.sigmoid(outputs)

        #convert probabilities into a binary assessement with 0.5 being the threshold, 
        predicted = (probabilities > 0.5).float()

        n_samples += labels.size[0] #this should represent the batch size

        # For each attribute, update correct and total counts
        for i in range(26):
            n_class_correct[i] += ((predicted[:, i] == labels[:, i])).sum().item()
            n_class_samples[i] += labels[:, i].size(0)
        
        if (i+1)%100 == 0:
            print(i, 'iterations have been completed')

    for i in range(26):
        accuracy = 100.0 * n_class_correct[i] / n_class_samples[i] if n_class_samples[i] > 0 else 0.0
        print(f"Attribute {i}: Accuracy = {accuracy:.2f}%")
'''