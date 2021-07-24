import torch
import torch.nn.functional as F
import torchvision
from modules.vqvae import *
from itertools import count
import matplotlib.pyplot as plt
from matplotlib import cm
import pickle
import random
from math import sqrt
import umap
import pandas as pd
import numpy as np



# we use this model for calculations if we are copying
# the weights from a model saved to disk
high_k = 3
low_k = 1024
class PARAMS:
    in_channels = 3                 # input image channels. 1 -> gray scale
    out_channels = 3                # output image channels. 1 -> gray scale
    blocks = [32, 64, 128, 256]     # convolutional block dims, the last number will be the codebook dim
    k = high_k                      # number of keys in the codebook
    lk = low_k
    
    w_code = 0.25

model = VQVAE2IN(PARAMS) # vqvae with instance norm (IN)


batch_size_test = 6


# demorms image data for plotting
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


opt = torch.optim.Adam(model.parameters(), lr=0.0001)


# here we will store data relating to vector quantization
# use these if you want to do dimensionality reduction
embeddings = []
indices_list = []
quantized_list = []
unquantized_list = []


# loading MNIST data
'''
test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


'''

# loading CelebA data
image_size = 256
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CelebA('/files/',
                              download=False,
                              transform=torchvision.transforms.Compose([
                                  torchvision.transforms.Resize(image_size),
                                  torchvision.transforms.CenterCrop(image_size),
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                       std=[0.5, 0.5, 0.5])
                              ])),
                                           batch_size=batch_size_test, shuffle=False)



# dictionaries storing which patches correspond to different indices
# i.e. patch_dict[14] stores all patches corresponding to index 14
# h_patch dict corresponds to the high level VQ section (vq_b in the model)
# l_patch dict corresponds to the low level VQ section (vq_t in the model)
h_width = 32
h_patch_dict = {}
for i in range(high_k):
    h_patch_dict[i] = []

l_width = 64   
l_patch_dict = {}
for i in range(low_k):
    l_patch_dict[i] = []


# get a specific patch of a specific image
def get_patch(data, image_number, patch_index, patch_number, patch_count):

    image = data[image_number]
    image = image.flatten(1)
   

    width = int(256 / sqrt(patch_count))
    patch = torch.zeros((3, width*width))

    
    for i in range(3):

        
        patch[i] = image[i][patch_number*width*width:(patch_number + 1)*width*width]

    patch = patch.unflatten(1, (width, width))
    return patch



# evaluate our model and print loss
# returns the average mse over all examples
def eval_model(test_model, input_loader, batch_size_test):

    total_mse = 0
    count_mse = 0
    test_model.eval()
    example_count = len(input_loader)
    examples = enumerate(input_loader)
    
    for step in range(example_count):

        batch_idx, (example_data, example_targets) = next(examples)

    
        # ir MNIST data, we must resize to 32x32 since MNIST is 28x28 (not powers of two)
        #example_data = F.interpolate(example_data, (32, 32))


        # vq_b is the output of high level VQ
        # vq_t is the output of low level VQ
        xr, vq_b, vq_t = test_model(example_data)

        
        code_loss = vq_b.loss + vq_t.loss
        loss_recon = F.mse_loss(example_data, xr)

        
        # collect information regarding high level quantizations

        '''
      
        unquantized = vq_b.unquantized
        unquantized = unquantized.reshape((6, 1024, 256))

        
        encoding_indices = vq_b.encoding_indices.flatten(1)#.encoding_indices.reshape((-1, 1024))

        embeddings.append(vq_b.embeddings.T)
        quantized = vq_b.quantized.reshape((6, 1024, 256))#.permute(0, 2, 1)
        
        # celeba
        for i in range(batch_size_test):

            for j in range(1024):
                indices_list.append(encoding_indices[i][j])
                quantized_list.append(quantized[i][j])
                unquantized_list.append(unquantized[i][j])
        '''
        
        '''
        # mnist
        for i in range(batch_size_test):
            
            indices_dict[example_targets[i].item()].append(encoding_indices[i])
            quantized_dict[example_targets[i].item()].append(quantized[i])
            unquantized_dict[example_targets[i].item()].append(unquantized[i])
        
        h_encoding_indices = vq_b.encoding_indices.flatten(1)

        '''
        '''
        h_encoding_indices = vq_b.encoding_indices.flatten(1)
        image_number = 0
        for image_indices in h_encoding_indices:

            
          
            patch_number = 0
            patch_count = len(image_indices)
            for index in image_indices:

                
                h_patch_dict[index.item()].append(get_patch(example_data, image_number, index, patch_number, patch_count))
                patch_number += 1

            image_number += 1
        '''
        
        '''
        fig = plt.figure()
        for i in range(6):
          plt.subplot(2,3,i+1)
          plt.tight_layout()
          plt.imshow(denorm(example_data[i].cpu().permute(1, 2, 0).data).numpy())
          #plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
          #plt.title("Ground Truth: {}".format(example_targets[i]))
          plt.xticks([])
          plt.yticks([])
        fig.show()
        '''

        
        
        
        


        # code for plotting input, output, vq layers of a batch      
        '''
        xr = xr.detach().numpy()

        
        fig2 = plt.figure()
        for i in range(6):
          plt.subplot(2,3,i+1)
          plt.tight_layout()

          plt.imshow(denorm(torch.from_numpy(xr)[i].cpu().permute(1, 2, 0).data).numpy())
          #plt.imshow(xr[i][0], cmap='gray', interpolation='none')
          #plt.title("Ground Truth: {}".format(example_targets[i]))
          plt.xticks([])
          plt.yticks([])
        fig2.show()

        # plot high level vector quantization

        fig3 = plt.figure()
        for i in range(6):
          plt.subplot(2,3,i+1)
          plt.tight_layout()

          #plt.imshow(denorm(torch.from_numpy(xr)[i].cpu().permute(1, 2, 0).data).numpy())
          plt.imshow(vq_b.encoding_indices[i].detach().numpy(), cmap='tab20', interpolation='none')
          #plt.title("Ground Truth: {}".format(example_targets[i]))
          plt.xticks([])
          plt.yticks([])
        fig3.show()
        

        # plot low level vector quantization
        fig4 = plt.figure()
        for i in range(6):
          plt.subplot(2,3,i+1)
          plt.tight_layout()

          #plt.imshow(denorm(torch.from_numpy(xr)[i].cpu().permute(1, 2, 0).data).numpy())
          plt.imshow(vq_t.encoding_indices[i].detach().numpy(), cmap='tab20', interpolation='none')
          #plt.title("Ground Truth: {}".format(example_targets[i]))
          plt.xticks([])
          plt.yticks([])
        fig4.show()
        
        '''
        
        loss_tot = loss_recon + PARAMS.w_code * code_loss
        print(f'>>> Step {step}, loss_recon={loss_recon.data}')
        total_mse += loss_recon.data
        count_mse += 1
        
    # here, indices_dict and quantized_dict contains all information
    # regarding VQ stage of model
    return total_mse / count_mse


# load model directly from file
model_path = '/files/CelebA/models/celeba_epochs_19_4.mdl'
new_model = torch.load(model_path, map_location=torch.device('cpu'))
#model.load_state_dict(new_model.state_dict()) # use this if we want to load weights instead
average_mse = eval_model(new_model, test_loader, batch_size_test)
        

# the following code relates to dimensionality reduction/visualization

# convert all unquantized tensors to numpy arrays
for i in range(len(unquantized_list)):
    unquantized_list[i] = unquantized_list[i].detach().numpy()


# obtain our list of embeddings
e_list = []
for i in range(64):
    e_list.append(e[i].detach().numpy())

# collect all vectors we want to perform dim reduction on 
k = unquantized_list + e_list
embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.7,
                      metric='correlation').fit_transform(k)


# we will colour different vector quantizations by the tab20 cmap
def get_colour(x):

    tab = plt.get_cmap('tab20')
    return tab(x)


# determine the colour for each vector
colours = []
for i in indices_list:
    colours.append(get_colour(i.item()))

plt.figure(figsize=(12,12))
plt.scatter(embedding[: len(k) - len(e_list), 0], embedding[: len(k) - len(e_list), 1],
            c=colours,
            edgecolor='none', 
            alpha=0.80, 
            s=10)


# add our codewords to the plot
for i in range(len(e_list)):
    plt.plot(embedding[len(k) - len(e_list) + i, 0], embedding[len(k) - len(e_list) + i , 1], marker='X', markersize=20, linewidth=30, c='black')

plt.axis('off');
plt.show()

