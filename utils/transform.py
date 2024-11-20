import torch
import torchvision.transforms as transforms
from PIL import Image

class ImageProcessor:
    def __init__(self, img_size=512):
        """
        Initializes the ImageProcessor with specified image size.
        """
        self.img_size = img_size

        # Preprocessing pipeline
        self.prep = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),  # Resize to specified dimensions
            transforms.ToTensor(),  # Convert image to tensor
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # Convert RGB to BGR
            transforms.Normalize(mean=[0.40760392, 0.45795686, 0.48501961],  # Subtract ImageNet mean
                                 std=[1, 1, 1]),
            transforms.Lambda(lambda x: x.mul_(255)),  # Scale values to 0-255
        ])

        # Postprocessing pipeline A
        self.postpa = transforms.Compose([
            transforms.Lambda(lambda x: x.mul_(1./255)),  # Scale values back to 0-1
            transforms.Normalize(mean=[-0.40760392, -0.45795686, -0.48501961],  # Add ImageNet mean
                                 std=[1, 1, 1]),
            transforms.Lambda(lambda x: x[torch.LongTensor([2, 1, 0])]),  # Convert BGR to RGB
        ])

        # Postprocessing pipeline B
        self.postpb = transforms.Compose([
            transforms.ToPILImage(),  # Convert tensor to PIL image
        ])

    def preprocess(self, image_path):
        """
        Preprocess an image from a given file path.
        :param image_path: Path to the image file.
        :return: Preprocessed image tensor.
        """
        image = Image.open(image_path).convert('RGB')
        return self.prep(image).unsqueeze(0)  # Add batch dimension

    def postprocess(self, tensor):
        """
        Postprocess a tensor to convert it into a displayable image.
        :param tensor: Input tensor to process.
        :return: Processed PIL image.
        """
        t = self.postpa(tensor.squeeze(0))  # Remove batch dimension and apply postpa
        t[t > 1] = 1  # Clip values >1 to 1
        t[t < 0] = 0  # Clip values <0 to 0
        return self.postpb(t)  # Convert tensor to PIL image
"""
# Example usage:
if __name__ == "__main__":
    processor = ImageProcessor(img_size=512)

    # Preprocess an image
    content_image_path = "dataset/content/Tuebingen_Neckarfront.jpg"
    preprocessed_image = processor.preprocess(content_image_path)
    print("Preprocessed Image Tensor:", preprocessed_image.size())

    # Postprocess the tensor (for testing)
    reconstructed_image = processor.postprocess(preprocessed_image)
    reconstructed_image.show()  # Display the image


from torchvision import transforms

class Transform:
    def __init__(self):
        self.transforms = {
            'vgg19': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'efficientnet': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]),
            'simplecnn': transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        }
    def get_transform(self, model_name):
        if model_name in self.transforms:
            return self.transforms[model_name]
        else:
            raise ValueError("nama model tidak sesuai")
"""