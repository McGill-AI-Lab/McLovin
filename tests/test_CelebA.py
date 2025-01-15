import pytest
import torch
import os
from dataclasses import dataclass
from typing import List
#from src.ml.image_description.models.conv_net import ConvNet
from src.ml.image_description.inference import load_model, process_image, get_attributes
from src.ml.image_description.description_generator import configure_generative_ai, generate_description
from src.ml.image_description.config import MODEL_PATH
from dotenv import load_dotenv

load_dotenv()

@dataclass
class FaceAttributes:
    physical_traits: List[str]
    description: str

@pytest.fixture(scope="module")
def model():
    # Ensure the model file is present
    assert os.path.exists(MODEL_PATH), f"Model file not found: {MODEL_PATH}"
    return load_model(MODEL_PATH)

def test_model_output_shape(model):
    image_tensor = process_image('helpers/sample_faces/suzy.jpg')
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0))
    assert output.shape == (1, 26), f"Expected output shape (1, 26), got {output.shape}"

def test_detect_attributes(model):
    image_tensor = process_image('helpers/sample_faces/suzy.jpg')
    traits = get_attributes(model, image_tensor)
    print("face traits are: ", traits)
    assert isinstance(traits, list)
    assert len(traits) > 0

@pytest.mark.skipif(not os.getenv("celeba_key"), reason="Needs Generative AI key to run")
def test_text_generation():
    configure_generative_ai()
    sample_traits = ["Male", "Bald", "Smiling"]
    description = generate_description(sample_traits)
    assert isinstance(description, str)
    assert len(description) > 0

def test_face_attributes_dataclass():
    # Example usage
    fa = FaceAttributes(
        physical_traits=["Male", "Bald", "Smiling"],
        description="He is a cheerful bald male with a friendly smile."
    )
    assert fa.physical_traits[0] == "Male"
    assert "bald" in fa.description.lower()
