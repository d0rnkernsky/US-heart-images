import torch
from torch import nn
from torchvision import transforms 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from common.dataset import USImagesDataset
import numpy as np
import json
import scipy.linalg

def get_FID(fake_features_all, real_features_all):
    mu_y = fake_features_all.mean(0)
    mu_x = real_features_all.mean(0)
    sigma_y = get_covariance(fake_features_all)
    sigma_x = get_covariance(real_features_all)
    
    with torch.no_grad():
         return (mu_x - mu_y).dot(mu_x - mu_y) + torch.trace(sigma_x) + torch.trace(sigma_y) - \
        2*torch.trace(matrix_sqrt(sigma_x @ sigma_y))

    
def get_covariance(features):
    return torch.Tensor(np.cov(features.detach().numpy(), rowvar=False))


def matrix_sqrt(x):
    y = x.cpu().detach().numpy()
    y = scipy.linalg.sqrtm(y)
    return torch.Tensor(y.real, device=x.device)

def show_tensor_images(image_tensor, num_images=5, size=(1, 100, 100)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in a uniform grid.
    '''
    image_unflat = image_tensor.detach().cpu().view(-1, *size)
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


def crop(image, new_shape):
    '''
    Function for cropping an image tensor: Given an image tensor and the new shape,
    crops to the center pixels.
    Parameters:
        image: image tensor of shape (batch size, channels, height, width)
        new_shape: a torch.Size object with the shape you want x to have
    '''
    middle_height = image.shape[2] // 2
    middle_width = image.shape[3] // 2
    starting_height = middle_height - new_shape[2] // 2
    final_height = starting_height + new_shape[2]
    starting_width = middle_width - new_shape[3] // 2
    final_width = starting_width + new_shape[3]
    cropped_image = image[:, :, starting_height:final_height, starting_width:final_width]    
    return cropped_image


def save_config(path, conf):
    with open(path, "w") as f:
        json.dump(conf, f)
        
        
def save_loss(name, loss):
    a = np.asarray(loss)
    np.savetxt(name, a, delimiter=",")