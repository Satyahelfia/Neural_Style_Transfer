import torch
from utils.getData import preprocess_image, postprocess_image
from models.vgg-19 import VGG19
from models.gram-matrix import GramMatrix
from models.loss import StyleLoss, ContentLoss

# Define paths
content_image_path = 'dataset/content/Tuebingen_Neckarfront.jpg'
style_image_path = 'dataset/style/vangogh_starry_night.jpg'

# Preprocess images
content_image = preprocess_image(content_image_path)
style_image = preprocess_image(style_image_path)

# Load model
vgg = VGG19()
vgg.eval()

# Define losses
content_loss = ContentLoss(target=content_image)
style_loss = StyleLoss(target=style_image)

# Training loop (simplified)
# ... Define optimizer, loss computation, and backward pass ...
print('Training complete!')
