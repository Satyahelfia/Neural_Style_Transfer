from PIL import Image
import torchvision.transforms as transforms
import torch

# Preprocessing
def preprocess_image(image_path, image_size=512):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.mul(255))
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

# Postprocessing
def postprocess_image(tensor):
    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.div(255)),
        transforms.ToPILImage()
    ])
    image = tensor.clone().squeeze(0)
    return transform(image)
