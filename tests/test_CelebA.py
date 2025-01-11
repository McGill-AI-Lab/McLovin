import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import torchvision
from torchvision import transforms

import google.generativeai as genai

from PIL import Image

from dotenv import load_dotenv
import os

#defining the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#The textual representation of the output for [26] tensor
value_of_attributes = ['5_o_Clock_Shadow',
                        'Bald',
                        'Bangs',
                        'Big_Lips',
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
                        'Young']
print(len(value_of_attributes))

# Define transformations for images
transform = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize images
    transforms.ToTensor(),  # Convert images to tensors
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize
])

im = Image.open('/Users/dorysong/Downloads/IMG_0991.jpg')

if im.mode != 'RGB':
    im = im.convert('RGB')

transformed_image = transform(im)

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

with torch.no_grad():

    attribute_assumption = [0 for i in range(26)]
    image = transformed_image.to(device)
    output = loaded_model(image)

    probabilities = torch.sigmoid(output)

    #convert probabilities into a binary assessement with 0.5 being the threshold, 
    predicted = (probabilities > 0.5).float()

    # Convert the attributes (a tensor) to a numpy array
    attributes = predicted.detach().cpu().numpy().tolist()
    the_attributes = attributes[0]

    user_phys_traits = []
    for index in range(len(the_attributes)):
        if value_of_attributes[index] == 'Male':
            user_phys_traits.append('Female' if the_attributes[index] == 0.0 else 'Male')
        else:
            if the_attributes[index] == 1.0:
                user_phys_traits.append(value_of_attributes[index])
    print(user_phys_traits)
    
    #Formatting the instructions for the text generator
    instructions = f'Write 1-3 concise sentences on the user\'s portrait with the following physical attributes: {user_phys_traits}'

    #Setting up the text generator
    load_dotenv()
    API = os.getenv("celeba_key")
    genai.configure(api_key=API)
    GenModel = genai.GenerativeModel("gemini-1.5-flash", system_instruction=instructions)

    #Generating bios
    response = GenModel.generate_content("Write the sentences",
                                
                                        generation_config=genai.types.GenerationConfig(
                                            max_output_tokens = 100,
                                            temperature= 1.0,
                                        ),
                                        )

    print(response.text)
    
