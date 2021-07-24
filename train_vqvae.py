import torch
import torch.nn.functional as F
import torchvision
from modules.vqvae import *

from itertools import count
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

class PARAMS:
    in_channels = 3                 # input image channels. 1 -> gray scale
    out_channels = 3                # output image channels. 1 -> gray scale
    blocks = [32, 64, 128, 256]     # convolutional block dims, the last number will be the codebook dim (b1, b2, b3, b4)
    k = 16                          # number of keys in the codebook 

    w_code = 0.25

model = VQVAE2IN(PARAMS) # vqvae with instance norm (IN)
opt = torch.optim.Adam(model.parameters(), lr=0.0001)

# we have to set paramaters before obtaining MNIST
n_epochs = 3
batch_size_train = 1
batch_size_test = 1000
learning_rate = 0.01
momentum = 0.5
log_interval = 10

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# obtain MNIST data

train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('/files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)


# uncomment to train ciphar10 data
'''
# obtain cifar 10 data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])


print("getting data")
trainset = torchvision.datasets.CIFAR10(root='/files/', train=True,
                                        download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                          shuffle=True)

testset = torchvision.datasets.CIFAR10(root='/files/', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''


# uncomment to train celebA data
'''
image_size = 256
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.CelebA('/files/',
                              download=False,
                              transform=torchvision.transforms.Compose([
                                  torchvision.transforms.Resize(image_size),
                                  torchvision.transforms.CenterCrop(image_size),
                                  torchvision.transforms.ToTensor(),
                                  torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                       std=[0.5, 0.5, 0.5])
                              ])),
                                           batch_size=batch_size_train, shuffle=True)
    
'''

#examples = enumerate(train_loader)
# batch_idx, (example_data, example_targets) = next(examples) 
# example data is [64, 1, 28, 28]    (training)
# example targets is [64]            (training)

#model = torch.load('/files/MNIST/models/MNIST_epochs_11_keys_16.mdl')


# first argument is initialized model
# second argument is our data as a Dataloader
# third argument is the total number of epochs
def train_model(training_model, input_loader, epoch_n):

    model.train()
 
    examples = enumerate(input_loader)
    
    for i in range(epoch_n):
        for j in range(len(input_loader)):


            batch_idx, (example_data, example_targets) = next(examples)

            # if 28x28 data, resize to 32x32 e.g. since MNIST is 28x28 (not powers of two)
            #example_data = F.interpolate(example_data, (32, 32))

        
            xr, vq_b, vq_t = training_model(example_data)
            code_loss = vq_b.loss + vq_t.loss
            loss_recon = F.mse_loss(example_data, xr)


            # uncomment if you'd like to plot input, output
            # note that different datasets may require different
            # denormalization, hence some lines being commented within
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
            
            
            xr = xr.detach().numpy()
            fig = plt.figure()
            for i in range(3):
              plt.subplot(2,3,i+1)
              plt.tight_layout()
              plt.imshow(xr[i][0], cmap='gray', interpolation='none')
              #plt.imshow(denorm(torch.from_numpy(xr)[i].cpu().permute(1, 2, 0).data).numpy())
              #plt.imshow(xr[i][0])
              plt.title("Ground Truth: {}".format(example_targets[i]))
              plt.xticks([])
              plt.yticks([])
            fig.show()
            
            '''
         
            loss_tot = loss_recon + PARAMS.w_code * code_loss
            
            opt.zero_grad()
            loss_tot.backward()
            opt.step()
            print(f'>>> Step {i}, loss_recon={loss_recon.data}')
        break
    return training_model

print("training...")
trained_model = train_model(model, train_loader, 3)
# here, the model has been trained. Lets save it
#torch.save(model, '/files/FashionMNIST/models/FashionMNIST_total_3') # example of saving a model to some directory
print("model saved")

