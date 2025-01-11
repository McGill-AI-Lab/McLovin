import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision
from torchvision import transforms
import google.generativeai as genai

from PIL import Image

# defining the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# The textual representation of the output for [26] tensor
value_of_attributes = [
    '5_o_Clock_Shadow',
    'Bald',
    'Bangs',          # Added missing comma here
    'Big_Lips',       # Added missing comma here
    'Big_Nose',
    'Black_Hair',
    'Blond_Hair',
    'Brown_Hair',
    'Bushy_Eyebrows',
    'Chubby',
    'Goatee',
    'Gray_Hair',
    'Heavy_Makeup',
    'High_Cheekbones',
    'Male',
    'Mustache',
    'Narrow_Eyes',
    'No_Beard',
    'Oval_Face',
    'Pale_Skin',
    'Pointy_Nose',
    'Sideburns',
    'Smiling',
    'Straight_Hair',
    'Wavy_Hair',
    'Young'
]

# Rest of your code remains the same
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, 3)
        self.fc1 = nn.Linear(128 * 30 * 30, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 26)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 30 * 30)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Load and process image
def process_image(image_path):
    im = Image.open(image_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return transform(im)

# Prediction function
def get_attributes(model, image_tensor):
    model.eval()
    with torch.no_grad():
        image = image_tensor.unsqueeze(0).to(device)
        output = model(image)
        probabilities = torch.sigmoid(output)
        predicted = (probabilities > 0.5).float()
        attributes = predicted.numpy()

        user_phys_traits = []
        for index in range(len(attributes[0])):
            if value_of_attributes[index] == 'Male':
                user_phys_traits.append('Female' if attributes[0][index] == 0 else 'Male')
            else:
                if attributes[0][index] == 1:
                    user_phys_traits.append(value_of_attributes[index])
        return user_phys_traits

# Main execution for testing the picture in utils/face_generator
if __name__ == "__main__":
    FILE = "model.pth"
    loaded_model = ConvNet().to(device)
    loaded_model.load_state_dict(torch.load(FILE, map_location=device, weights_only=True))

    transformed_image = process_image('utils/face_generator/michael_jordan.jpg')
    traits = get_attributes(loaded_model, transformed_image)
    print("Detected traits:", traits)
