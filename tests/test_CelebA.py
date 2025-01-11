import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import os
from dataclasses import dataclass
from typing import List

@dataclass
class FaceAttributes:
    physical_traits: List[str]
    description: str

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

class FaceAttributeDetector:
    def __init__(self, model_path: str):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.transform = self._get_transform()
        self.attributes = [
            '5_o_Clock_Shadow', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
            'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
            'Chubby', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
            'Male', 'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face',
            'Pale_Skin', 'Pointy_Nose', 'Sideburns', 'Smiling', 'Straight_Hair',
            'Wavy_Hair', 'Young'
        ]

    def _load_model(self, model_path: str) -> nn.Module:
        model = ConvNet().to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        model.eval()
        return model

    def _get_transform(self):
        return transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def process_image(self, image_path: str) -> torch.Tensor:
        image = Image.open(image_path)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return self.transform(image)

    def detect_attributes(self, image_tensor: torch.Tensor) -> List[str]:
        with torch.no_grad():
            image = image_tensor.unsqueeze(0).to(self.device)
            output = self.model(image)
            probabilities = torch.sigmoid(output)
            predicted = (probabilities > 0.5).float()
            attributes = predicted.detach().cpu().numpy().tolist()[0]

            detected_traits = []
            for idx, is_present in enumerate(attributes):
                if self.attributes[idx] == 'Male':
                    detected_traits.append('Female' if is_present == 0.0 else 'Male')
                elif is_present == 1.0:
                    detected_traits.append(self.attributes[idx])

            return detected_traits

class TextGenerator:
    def __init__(self, api_key_env: str = "celeba_key"):
        load_dotenv()
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"API key not found in environment variable: {api_key_env}")

        genai.configure(api_key=api_key)


    def generate_description(self, traits: List[str]) -> str:
        # initialize the model here because trait is necessary
        instructions = f'Write 1-3 concise sentences on the user\'s portrait with the following physical attributes: {traits}'
        self.model = genai.GenerativeModel("gemini-1.5-flash", system_instruction=instructions)

        response = self.model.generate_content(
            "Write the sentences",
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=100,
                temperature=1.0,
            )
        )
        return response.text

class FaceProcessor:
    def __init__(self, model_path: str):
        self.detector = FaceAttributeDetector(model_path)
        self.generator = TextGenerator()

    def process_face(self, image_path: str) -> FaceAttributes:
        image_tensor = self.detector.process_image(image_path)
        traits = self.detector.detect_attributes(image_tensor)
        description = self.generator.generate_description(traits)
        return FaceAttributes(physical_traits=traits, description=description)

def test_face_processing():
    processor = FaceProcessor("model.pth")

    # Test image processing
    result = processor.process_face('utils/face_generator/michael_jordan.jpg')

    # Assertions
    assert isinstance(result, FaceAttributes)
    assert isinstance(result.physical_traits, list)
    assert len(result.physical_traits) > 0
    assert isinstance(result.description, str)
    assert len(result.description) > 0

    print("Face Attributes:", result.physical_traits)
    print("Generated Description:", result.description)

def test_model_output_shape():
    detector = FaceAttributeDetector("model.pth")
    image_tensor = detector.process_image('utils/face_generator/michael_jordan.jpg')

    with torch.no_grad():
        image = image_tensor.unsqueeze(0).to(detector.device)
        output = detector.model(image)

    assert output.shape == (1, 26), f"Expected output shape (1, 26), got {output.shape}"

def test_attribute_list_length():
    detector = FaceAttributeDetector("model.pth")
    assert len(detector.attributes) == 26, f"Expected 26 attributes, got {len(detector.attributes)}"

if __name__ == "__main__":
    pytest.main([__file__])
