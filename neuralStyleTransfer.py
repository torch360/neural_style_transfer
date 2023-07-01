import torch
import torch.nn as nn
import torchvision.models as models
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import torchvision.utils as utils

class VGGNST(nn.Module):

    def __init__(self):
        super(VGGNST, self).__init__()
        self.model = models.vgg19(pretrained=True).features[:37]
        for i in [4, 9, 18, 27]:
            self.model[i] = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        samples = []
        for i in range(37):
            x = self.model[i](x)
            if i in [5, 10, 19, 28]:
                samples.append(x)
        return samples


def load_img(path, size):
    transform = transforms.Compose([
        transforms.Resize((size,size)),
        transforms.ToTensor(),
        ])
    img = Image.open(path)
    img = transform(img).unsqueeze(0)
    return img

def train_style(epochs, alpha, beta, optimizer, generated_image, content_image, style_image):
    for i in range(epochs):
        generated_features = model(generated_image)
        content_features = model(content_image)
        style_features = model(style_image)

        c_loss = 0
        s_loss = 0

        for generated_feature, content_feature, style_feature in zip(generated_features, content_features, style_features):
            batch_size, n_feature_maps, height, width = generated_feature.size()

            c_loss += torch.mean((generated_feature - content_feature)**2)  # this is the minimizing differences


            Gram = torch.mm(generated_feature.view(batch_size*n_feature_maps, height*width),
                            generated_feature.view(batch_size*n_feature_maps, height*width).t())
            A = torch.mm(style_feature.view(batch_size*n_feature_maps, height*width),
                         style_feature.view(batch_size*n_feature_maps, height*width).t())


            E_l = (Gram-A)**2
            w_l = 1/5
            s_loss += torch.mean(E_l*w_l)

        total_loss = alpha*c_loss + beta*s_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        print(i)
    return generated_image


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
content_img = load_img("./images/moraine.jpeg", 512).to(device)
style_img = load_img("./images/stars.jpeg", 512).to(device)

model = VGGNST().to(device)

for param in model.parameters():
    param.requires_grad = False

generated_init = content_img.clone().requires_grad_(True)

epochs = 300

lr = 1e-2
alpha = 1
beta = 3

optimizer = optim.Adam([generated_init], lr=lr)

generated_image = train_style(epochs=epochs,
                              alpha=alpha,
                              beta=beta,
                              optimizer=optimizer,
                              generated_image=generated_init,
                              content_image=content_img,
                              style_image=style_img)
utils.save_image(generated_image, f"./gen_images/gen.png")


