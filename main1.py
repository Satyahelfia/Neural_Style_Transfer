import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.transform import ImageProcessor
from models.vgg import VGG
from models.GramMatrix import GramMatrix, GramMSELoss
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Inisialisasi ImageProcessor
img_processor = ImageProcessor(img_size=512)

# Load style dan content image
style_image_path = "dataset/style/vangogh_starry_night.jpg"
content_image_path = "dataset/content/Tuebingen_Neckarfront.jpg"

style_image = img_processor.preprocess(style_image_path)
content_image = img_processor.preprocess(content_image_path)

# Pindahkan ke GPU jika tersedia
style_image = style_image.to(device)
content_image = content_image.to(device)

# Tensor untuk optimasi, pindahkan juga ke GPU
opt_img = Variable(content_image.data.clone(), requires_grad=True).to(device)

# Load VGG model dan pindahkan ke GPU
vgg = VGG().to(device)

# Definisikan layer, fungsi loss, dan target
style_layers = ['r11', 'r21', 'r31', 'r41', 'r51']
content_layers = ['r42']
loss_layers = style_layers + content_layers

loss_fns = [GramMSELoss().to(device)] * len(style_layers) + [nn.MSELoss().to(device)] * len(content_layers)

# Bobot untuk style dan content loss
style_weights = [1e3 / n ** 2 for n in [64, 128, 256, 512, 512]]
content_weights = [1e0]
weights = style_weights + content_weights

# Hitung target optimasi
style_targets = [GramMatrix()(A).detach().to(device) for A in vgg(style_image, style_layers)]
content_targets = [A.detach().to(device) for A in vgg(content_image, content_layers)]
targets = style_targets + content_targets

# Optimizer
optimizer = optim.LBFGS([opt_img])
max_iter = 500
show_iter = 50
n_iter = [0]

# Loop optimasi
while n_iter[0] <= max_iter:
    def closure():
        optimizer.zero_grad()
        out = vgg(opt_img, loss_layers)
        layer_losses = [weights[a] * loss_fns[a](A, targets[a]) for a, A in enumerate(out)]
        loss = sum(layer_losses)
        loss.backward()
        n_iter[0] += 1
        if n_iter[0] % show_iter == 0:
            print(f"Iteration: {n_iter[0]}, Loss: {loss.item()}")
        return loss

    optimizer.step(closure)

# Postprocess dan simpan hasil
result_image = img_processor.postprocess(opt_img.data[0].cpu())

# Menyimpan gambar hasil ke folder output/generate_vgg
output_folder = "output/generate_vgg"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Simpan gambar hasil optimasi
result_image_path = os.path.join(output_folder, "style_transferred_image.jpg")
result_image.save(result_image_path)
print(f"Image saved to {result_image_path}")

# Tampilkan gambar hasil
result_image.show()
