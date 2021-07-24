import torch
import torch.nn as nn
import torch.nn.functional as F
from .vq import VectorQuantizer2D, VectorQuantizer1D
from .conv_blocks import UpBlock, DownBlock, UpBlock3D, DownBlock3D
from modules.transformer import *
import matplotlib.pyplot as plt

        
class VQVAE2(nn.Module):
    def __init__(self, params):
        '''
        class PARAMS:
            in_channels = 3
            out_channels = 3
            blocks = [32, 64, 128, 256]
            k = 256
        '''
        super().__init__()
        in_channels, out_channels, blocks, k = params.in_channels, params.out_channels, params.blocks, params.k
        b1, b2, b3, b4 = blocks

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock(b1, b2)
        self.down2 = DownBlock(b2, b3)
        self.down3 = DownBlock(b3, b4)

        
        self.vq_b = VectorQuantizer2D(b4, k)
        self.vq_t_conv = nn.Conv2d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer2D(b4, k)

        self.up1 = UpBlock(b4, b4)
        self.up1vq = UpBlock(b4, b4)
        self.up2 = UpBlock(b4 * 2, b3)
        self.up3 = UpBlock(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv2d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, out_channels, 1),
        )

    def encode(self, x):
        x = self.block(x)
        x = self.down1(x)

        # transformer encoder here?
        
        x_b = self.down2(x)
        x = self.down3(x_b)

        # or maybe here?
        
        vq_b = self.vq_b(x)
        vqb_qantized = vq_b.quantized
    
        x = self.up1(vqb_qantized)
        x = torch.cat([x, x_b], 1)
        x = self.vq_t_conv(x)
        vq_t = self.vq_t(x)
        vq_t_qantized = vq_t.quantized

        return vq_b, vq_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized):
        vqb_up = self.up1vq(vqb_qantized)
        x = torch.cat([vqb_up, vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        vq_b, vq_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized, vq_t.quantized)
        return x, vq_b, vq_t


class VQVAE2IN(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels, out_channels, blocks, k = params.in_channels, params.out_channels, params.blocks, params.k
        b1, b2, b3, b4 = blocks

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock(b1, b2)
        self.down2 = DownBlock(b2, b3)
        self.down3 = DownBlock(b3, b4)

        self.in_b = nn.InstanceNorm2d(b4)
        self.in_m = nn.InstanceNorm2d(b3)

        self.vq_b = VectorQuantizer2D(b4, k)
        self.vq_t_conv = nn.Conv2d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer2D(b4, k)

        self.up1 = UpBlock(b4, b4)
        self.up1vq = UpBlock(b4, b4)
        self.up2 = UpBlock(b4 * 2, b3)
        self.up3 = UpBlock(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv2d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, out_channels, 1),
        )

    def encode(self, x):

        
        x = self.block(x)
        x = self.down1(x)
        x_b = self.down2(x)

        x = self.down3(x_b)
      
        vq_b = self.vq_b(self.in_b(x))
        vqb_qantized = vq_b.quantized
        x = self.up1(vqb_qantized)


        print(x.shape)
        print(x_b.shape)
        x = torch.cat([x, x_b], 1)
        x = self.vq_t_conv(x)
        vq_t = self.vq_t(self.in_m(x))
        vq_t_qantized = vq_t.quantized
        
        return vq_b, vq_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized):
        
        vqb_up = self.up1vq(vqb_qantized)
        x = torch.cat([vqb_up, vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        vq_b, vq_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized, vq_t.quantized)
        return x, vq_b, vq_t


# changes the number of blocks in vq layers
# we add down layers to further reduce the image
# to a smaller number of blocks/patches
class BLOCK_VQVAE2IN(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels, out_channels, blocks, k = params.in_channels, params.out_channels, params.blocks, params.k
        b1, b2, b3, b4 = blocks

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock(b1, b2)
        self.down2 = DownBlock(b2, b3)
        self.down3 = DownBlock(b3, b4)
        
        self.down4 = DownBlock(b4, b4)
        self.down5 = DownBlock(b4, b4)

        # only needed for 256x256 images
        # use these layers in encoder and
        # 256x256 images will become one vector
        self.down6 = DownBlock(b4, b4)
        self.down7 = DownBlock(b4, b4)
        self.down8 = DownBlock(b4, b4)
        

        self.in_b = nn.InstanceNorm2d(b4)
        self.in_m = nn.InstanceNorm2d(b3)

        self.vq_b = VectorQuantizer2D(b4, k)
        self.vq_t_conv = nn.Conv2d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer2D(b4, k)

        self.up1 = UpBlock(b4, b4)
        self.up1a = UpBlock(b4, b4)
        self.up1b = UpBlock(b4, b4)

        # only needed for 256x256 images
        self.up1c = UpBlock(b4, b4)
        self.up1d = UpBlock(b4, b4)
        self.up1e = UpBlock(b4, b4)
        
        self.up1vq = UpBlock(b4, b4)
        self.up1avq = UpBlock(b4, b4)
        self.up1bvq = UpBlock(b4, b4)

        # only need these for 256x256 images
        self.up1cvq = UpBlock(b4, b4)
        self.up1dvq = UpBlock(b4, b4)
        self.up1evq = UpBlock(b4, b4)

        self.up2 = UpBlock(b4 * 2, b3)
        self.up3 = UpBlock(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv2d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, out_channels, 1),
        )

    def encode(self, x):

        
        x = self.block(x)

        
        fig = plt.figure()

        
        x = self.down1(x)
        x_b = self.down2(x)

        x = self.down3(x_b)
        x = self.down4(x)
        x = self.down5(x)

        # these are only needed for 256x256 images
        #x = self.down6(x)
        #x = self.down7(x)
        #x = self.down8(x)
        
        # don't need instance norm if only one element
        # if just one element, change to ...self.vq_b(x)
        vq_b = self.vq_b(self.in_b(x))
        vqb_qantized = vq_b.quantized
        x = self.up1(vqb_qantized)
        x = self.up1a(x)
        x = self.up1b(x)

        # only need these for 256x256 images
        #x = self.up1c(x)
        #x = self.up1d(x)
        #x = self.up1e(x)
        
        x = torch.cat([x, x_b], 1)
        x = self.vq_t_conv(x)
        vq_t = self.vq_t(self.in_m(x))
        vq_t_qantized = vq_t.quantized
        
        return vq_b, vq_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized):
        
        vqb_up = self.up1vq(vqb_qantized)
        vqb_up = self.up1avq(vqb_up)
        vqb_up = self.up1bvq(vqb_up)

        # only need these for 256x256 images
        #vqb_up = self.up1cvq(vqb_up)
        #vqb_up = self.up1dvq(vqb_up)
        #vqb_up = self.up1dvq(vqb_up)
        
        x = torch.cat([vqb_up, vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        vq_b, vq_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized, vq_t.quantized)
        return x, vq_b, vq_t



# allows us to change the codebook size of both VQ stages,
# params lk and hk refer to low VQ key number and high VQ key number
class DUAL_VQVAE2IN(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels, out_channels, blocks, lk, hk = params.in_channels, params.out_channels, params.blocks, params.k, params.lk
        b1, b2, b3, b4 = blocks

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock(b1, b2)
        self.down2 = DownBlock(b2, b3)
        self.down3 = DownBlock(b3, b4)

        self.in_b = nn.InstanceNorm2d(b4)
        self.in_m = nn.InstanceNorm2d(b3)

        self.vq_b = VectorQuantizer2D(b4, hk)
        self.vq_t_conv = nn.Conv2d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer2D(b4, lk)

        self.up1 = UpBlock(b4, b4)
        self.up1vq = UpBlock(b4, b4)
        self.up2 = UpBlock(b4 * 2, b3)
        self.up3 = UpBlock(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv2d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, out_channels, 1),
        )

    def encode(self, x):

        
        x = self.block(x)
        x = self.down1(x)
        x_b = self.down2(x)
        x = self.down3(x_b)
        vq_b = self.vq_b(self.in_b(x))
        vqb_qantized = vq_b.quantized
        x = self.up1(vqb_qantized)

        x = torch.cat([x, x_b], 1)
        x = self.vq_t_conv(x)
        vq_t = self.vq_t(self.in_m(x))
        vq_t_qantized = vq_t.quantized
        
        return vq_b, vq_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized):
        
        vqb_up = self.up1vq(vqb_qantized)
        x = torch.cat([vqb_up, vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        vq_b, vq_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized, vq_t.quantized)
        return x, vq_b, vq_t


class VQVAE2IN3D(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels, out_channels, blocks, k = params.in_channels, params.out_channels, params.blocks, params.k
        b1, b2, b3, b4 = blocks

        self.block = nn.Sequential(
            nn.Conv3d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock3D(b1, b2)
        self.down2 = DownBlock3D(b2, b3)
        self.down3 = DownBlock3D(b3, b4)

        self.in_b = nn.InstanceNorm3d(b4)
        self.in_m = nn.InstanceNorm3d(b3)

        self.vq_b = VectorQuantizer1D(b4, k)
        self.vq_t_conv = nn.Conv3d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer1D(b4, k)

        self.up1 = UpBlock3D(b4, b4)
        self.up1vq = UpBlock3D(b4, b4)
        self.up2 = UpBlock3D(b4 * 2, b3)
        self.up3 = UpBlock3D(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv3d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv3d(b1, out_channels, 1),
        )

    def encode(self, x):
        x = self.block(x)
        x = self.down1(x)
        x_b = self.down2(x)
        x = self.down3(x_b)
        shape_b = x.shape
        vq_b = self.vq_b(self.in_b(x).flatten(2))
        vqb_qantized = vq_b.quantized.reshape(shape_b)

        x = self.up1(vqb_qantized)
        x = torch.cat([F.interpolate(x, size=x_b.shape[2:]), x_b], 1)
        x = self.vq_t_conv(x)
        
        shape_t = x.shape
        vq_t = self.vq_t(self.in_m(x).flatten(2))
        vq_t_qantized = vq_t.quantized.reshape(shape_t)
        return vq_b, vq_t, shape_b, shape_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized, original_dim=None):
        vqb_up = self.up1vq(vqb_qantized)
        x = torch.cat([F.interpolate(vqb_up, size=vq_t_qantized.shape[2:]), vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = F.interpolate(x, size=original_dim)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        original_dim = x.shape[2:]
        vq_b, vq_t, shape_b, shape_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized.reshape(shape_b), vq_t.quantized.reshape(shape_t), original_dim=original_dim)
        return x, vq_b, vq_t

# "early transformer VQ-VAE"
# transformer encoder layer is put in the model before both VQ layers
class ET_VQVAE2IN(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels, out_channels, blocks, k = params.in_channels, params.out_channels, params.blocks, params.k
        b1, b2, b3, b4 = blocks

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock(b1, b2)
        self.down2 = DownBlock(b2, b3)
        self.down3 = DownBlock(b3, b4)


        self.t_encoder = TransformerEncoder(enc_seqlen=16*16, d_model=64, nhead=8,
                                            num_encoder_layers=1,
                                            dim_feedforward=b2)
        
        self.in_b = nn.InstanceNorm2d(b4)
        self.in_m = nn.InstanceNorm2d(b3)

        self.vq_b = VectorQuantizer2D(b4, k)
        self.vq_t_conv = nn.Conv2d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer2D(b4, k)

        self.up1 = UpBlock(b4, b4)
        self.up1vq = UpBlock(b4, b4)
        self.up2 = UpBlock(b4 * 2, b3)
        self.up3 = UpBlock(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv2d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, out_channels, 1),
        )

    def encode(self, x):

        
        x = self.block(x)
        x = self.down1(x)

        # do transformer encoding here(early)

        # we must flatten the image for the transformer encoder
        x = x.flatten(2)
        x = self.t_encoder(x, mask=None)

        # output will be first element of tuple
        x = x[0]

        # now we must unflatten the output
        x = x.unflatten(2, (16, 16))

        
        
        x_b = self.down2(x)
        x = self.down3(x_b)

    
        
        vq_b = self.vq_b(self.in_b(x))
        vqb_qantized = vq_b.quantized
        x = self.up1(vqb_qantized)

        x = torch.cat([x, x_b], 1)
        x = self.vq_t_conv(x)
        vq_t = self.vq_t(self.in_m(x))
        vq_t_qantized = vq_t.quantized
        
        return vq_b, vq_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized):
        
        vqb_up = self.up1vq(vqb_qantized)
        x = torch.cat([vqb_up, vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        vq_b, vq_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized, vq_t.quantized)
        return x, vq_b, vq_t


# "late transformer VQ-VAE" with block
# LT-VQVAE with same properties as block_vqvae
class LT_BLOCK_VQVAE2IN(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels, out_channels, blocks, k = params.in_channels, params.out_channels, params.blocks, params.k
        b1, b2, b3, b4 = blocks

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock(b1, b2)
        self.down2 = DownBlock(b2, b3)
        self.down3 = DownBlock(b3, b4)
        
        self.down4 = DownBlock(b4, b4)
        self.down5 = DownBlock(b4, b4)

        self.t_encoder = TransformerEncoder(enc_seqlen=64, d_model=256, nhead=8,
                                            num_encoder_layers=3,
                                            dim_feedforward=b4)
        # only needed for 256x256 images
        #self.down6 = DownBlock(b4, b4)
        #self.down7 = DownBlock(b4, b4)
        #elf.down8 = DownBlock(b4, b4)
        

        self.in_b = nn.InstanceNorm2d(b4)
        self.in_m = nn.InstanceNorm2d(b3)

        self.vq_b = VectorQuantizer2D(b4, k)
        self.vq_t_conv = nn.Conv2d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer2D(b4, k)

        self.up1 = UpBlock(b4, b4)
        self.up1a = UpBlock(b4, b4)
        self.up1b = UpBlock(b4, b4)

        # only needed for 256x256 images
        #self.up1c = UpBlock(b4, b4)
        #self.up1d = UpBlock(b4, b4)
        #self.up1e = UpBlock(b4, b4)
        
        self.up1vq = UpBlock(b4, b4)
        self.up1avq = UpBlock(b4, b4)
        self.up1bvq = UpBlock(b4, b4)

        # only need these for 256x256 images
        #self.up1cvq = UpBlock(b4, b4)
        #self.up1dvq = UpBlock(b4, b4)
        #self.up1evq = UpBlock(b4, b4)

        self.up2 = UpBlock(b4 * 2, b3)
        self.up3 = UpBlock(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv2d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, out_channels, 1),
        )

    def encode(self, x):

        
        x = self.block(x)

        
        fig = plt.figure()

        
        x = self.down1(x)
        x_b = self.down2(x)

        x = self.down3(x_b)
        x = self.down4(x)
        x = self.down5(x)

        x = x.flatten(2)
        print(x.shape)
        x = self.t_encoder(x, mask=None)

        # output will be first element of tuple
        x = x[0]
        
        # now we must unflatten the output
        x = x.unflatten(2, (8, 8))
       
      
        # these are only needed for 256x256 images
        #x = self.down6(x)
        #x = self.down7(x)
        #x = self.down8(x)
        
        # don't need instance norm if only one element
        vq_b = self.vq_b(self.in_b(x))
        vqb_qantized = vq_b.quantized
        x = self.up1(vqb_qantized)
        x = self.up1a(x)
        x = self.up1b(x)

        # only need these for 256x256 images
        #x = self.up1c(x)
        #x = self.up1d(x)
        #x = self.up1e(x)
        
        x = torch.cat([x, x_b], 1)
        x = self.vq_t_conv(x)
        vq_t = self.vq_t(self.in_m(x))
        vq_t_qantized = vq_t.quantized
        
        return vq_b, vq_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized):
        
        vqb_up = self.up1vq(vqb_qantized)
        vqb_up = self.up1avq(vqb_up)
        vqb_up = self.up1bvq(vqb_up)

        # only need these for 256x256 images
        #vqb_up = self.up1cvq(vqb_up)
        #vqb_up = self.up1dvq(vqb_up)
        #vqb_up = self.up1dvq(vqb_up)
        
        x = torch.cat([vqb_up, vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        vq_b, vq_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized, vq_t.quantized)
        return x, vq_b, vq_t


# "late transformer VQ-VAE"
# we add a transformer encoder layer just before the high level VQ section
class LT_VQVAE2IN(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels, out_channels, blocks, k = params.in_channels, params.out_channels, params.blocks, params.k
        b1, b2, b3, b4 = blocks

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock(b1, b2)
        self.down2 = DownBlock(b2, b3)
        self.down3 = DownBlock(b3, b4)

      
        self.t_encoder = TransformerEncoder(enc_seqlen=1024, d_model=256, nhead=8,
                                            num_encoder_layers=3,
                                            dim_feedforward=b4)
        
        self.in_b = nn.InstanceNorm2d(b4)
        self.in_m = nn.InstanceNorm2d(b3)

        self.vq_b = VectorQuantizer2D(b4, k)
        self.vq_t_conv = nn.Conv2d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer2D(b4, k)

        self.up1 = UpBlock(b4, b4)
        self.up1vq = UpBlock(b4, b4)
        self.up2 = UpBlock(b4 * 2, b3)
        self.up3 = UpBlock(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv2d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, out_channels, 1),
        )

    def encode(self, x):

        
        x = self.block(x)
        x = self.down1(x)
        x_b = self.down2(x)
        x = self.down3(x_b)

        # do transformer encoding here

        # we must flatten the image for the transformer encoder
        x = x.flatten(2)
        print(x.shape)
        x = self.t_encoder(x, mask=None)

        # output will be first element of tuple
        x = x[0]
        
        # now we must unflatten the output
        x = x.unflatten(2, (32, 32))
        
        vq_b = self.vq_b(self.in_b(x))
        vqb_qantized = vq_b.quantized
        x = self.up1(vqb_qantized)

        x = torch.cat([x, x_b], 1)
        x = self.vq_t_conv(x)
        vq_t = self.vq_t(self.in_m(x))
        vq_t_qantized = vq_t.quantized
        
        return vq_b, vq_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized):
        
        vqb_up = self.up1vq(vqb_qantized)
        x = torch.cat([vqb_up, vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        vq_b, vq_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized, vq_t.quantized)
        return x, vq_b, vq_t



class ONE_VQVAE2IN(nn.Module):
    def __init__(self, params):
        super().__init__()
        in_channels, out_channels, blocks, k = params.in_channels, params.out_channels, params.blocks, params.k
        b1, b2, b3, b4 = blocks

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, b1, 3, padding=1),
            nn.LeakyReLU()
        )

        self.down1 = DownBlock(b1, b2)
        self.down2 = DownBlock(b2, b3)
        self.down3 = DownBlock(b3, b4)

      
        #self.t_encoder = Transformer(enc_seqlen=16, dec_seqlen=1)

        self.encoder = TransformerEncoder(
            enc_seqlen=16,         # how many dims if flattened
            d_model=256,                # performer / transformer dimension
            nhead=8,
            num_encoder_layers=3,       # how many repeats
            dim_feedforward=1024
        )

        self.ff = nn.Linear(16*256*16, 256*16)
          
        self.in_b = nn.InstanceNorm2d(b4)
        self.in_m = nn.InstanceNorm2d(b3)

    
        self.vq_b = VectorQuantizer2D(b4, k)
        self.vq_t_conv = nn.Conv2d(b3 + b4, b4, 1)
        self.vq_t = VectorQuantizer2D(b4, k)

        self.up1 = UpBlock(b4, b4)
        self.up1a = UpBlock(b4, b4)
        self.up1b = UpBlock(b4, b4)
        
        self.up1vq = UpBlock(b4, b4)
        self.up2vq = UpBlock(b4, b4)
        self.up3vq = UpBlock(b4, b4)
        
        self.up2 = UpBlock(b4 * 2, b3)
        self.up3 = UpBlock(b3, b2)
        

        self.out = nn.Sequential(
            nn.Conv2d(b2, b1, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(b1, out_channels, 1),
        )

    def encode(self, x):

        
        x = self.block(x)
        x = self.down1(x)
        x_b = self.down2(x)
        x = self.down3(x_b)
        
        # do transformer encoding here
       
        # we must flatten the image for the transformer encoder
        x = x.flatten(2)
        print("encoder in: ", str(x.shape))
        query = x
        
        x, e = self.encoder(x, mask=None)

        combined_out = torch.cat((x, e), dim=0)

        x = self.ff(combined_out.flatten(0))
        x = x.unflatten(0, (16, 256, 1))
     
        print("encoder out: ", str(x.shape))
       
        # now we must unflatten the output
        x = x.unflatten(2, (1, 1))
        print("x shape: ", str(x.shape))
        
        vq_b = self.vq_b(x)
        vqb_qantized = vq_b.quantized

        
        x = self.up1(vqb_qantized)
        x = self.up1a(x)
        x = self.up1b(x)

    
        x = torch.cat([x, x_b], 1)

        
        x = self.vq_t_conv(x)
        
       
        vq_t = self.vq_t(self.in_m(x))
        vq_t_qantized = vq_t.quantized
        
        return vq_b, vq_t
        # vqb_qantized, vq_t_qantized, vq_b.loss + vq_t.loss, vq_b.encoding_indices,  vq_t.encoding_indices

    def decode(self, vqb_qantized, vq_t_qantized):
        
        vqb_up = self.up1vq(vqb_qantized)
        vqb_up = self.up2vq(vqb_up)
        vqb_up = self.up3vq(vqb_up)
        print(vqb_up.shape)
        print(vq_t_qantized.shape)

        
        x = torch.cat([vqb_up, vq_t_qantized], 1)
        x = self.up2(x)
        x = self.up3(x)
        x = self.out(x)
        return x

    def decode_logits_or_indices(self, pb, pt):
        '''
        Decode logits (bs, classes, w, h) or indices (bs, w, h)
        '''
        if len(pb.shape) == 4:
            pb = pb.argmax(1)
        if len(pt.shape) == 4:
            pt = pt.argmax(1)
        vqb_qantized = self.vq_b.quantize_indices(pb)
        vq_t_qantized = self.vq_t.quantize_indices(pt)
        return self.decode(vqb_qantized, vq_t_qantized)

    def forward(self, x):
        vq_b, vq_t = self.encode(x)
        # vqb_qantized, vq_t_qantized, loss, idx_b, idx_t = self.encode(x)
        x = self.decode(vq_b.quantized, vq_t.quantized)
        return x, vq_b, vq_t


