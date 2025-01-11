import torch
import torch.nn.functional as F
from PIL import Image
from config import DEVICE, MODEL_PATH
from models.conv_net import ConvNet

# The textual representation of the output for [26] tensor
VALUE_OF_ATTRIBUTES = [
    '5_o_Clock_Shadow', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
    'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Chubby', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
    'Pale_Skin', 'Pointy_Nose', 'Sideburns', 'Smiling', 'Straight_Hair',
    'Wavy_Hair', 'Young'
]

# Simple transform (same as the training transform)
from torchvision import transforms
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

def load_model(path=MODEL_PATH):
    model = ConvNet(num_classes=26).to(DEVICE)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.eval()
    return model

def process_image(image_path):
    im = Image.open(image_path)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    return transform(im)

def get_attributes(model, image_tensor):
    with torch.no_grad():
        image = image_tensor.unsqueeze(0).to(DEVICE)
        output = model(image)
        probabilities = torch.sigmoid(output)
        predicted = (probabilities > 0.5).float()
        attributes = predicted.cpu().numpy()

    user_phys_traits = []
    for index in range(len(attributes[0])):
        # Special case for 'Male' -> 'Female'
        if VALUE_OF_ATTRIBUTES[index] == 'Male':
            user_phys_traits.append('Female' if attributes[0][index] == 0 else 'Male')
        else:
            if attributes[0][index] == 1:
                user_phys_traits.append(VALUE_OF_ATTRIBUTES[index])

    return user_phys_traits

if __name__ == "__main__":
    model = load_model()
    transformed_image = process_image('helpers/face_generator/suzy.jpg')
    print(transformed_image)
    # debug
    #im = Image.open('helpers/face_generator/suzy.jpg')
    #im.show()

    traits = get_attributes(model, transformed_image)
    print("Detected traits:", traits)
