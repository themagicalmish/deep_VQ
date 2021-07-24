import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
import random

class VectorQuantizer1D(nn.Module):
    

    Output = namedtuple('Output', ('quantized', 'loss', 'one_hot', 'encoding_indices', 'distances', 'unquantized', 'embeddings'))
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.embeddings = nn.Parameter(torch.empty(embedding_dim, num_embeddings))
        nn.init.xavier_uniform_(self.embeddings)
        

    def quantize_indices(self, indices, output_img=True):

        
        encoding_indices = indices.long().flatten(0)
        quantized = torch.index_select(self.embeddings.T, 0, encoding_indices)
        return quantized.reshape(indices.shape + (self.embedding_dim,))
        

    def forward(self, x):
          
        seq_len = x.shape[2]

        bs = x.shape[0]
        inputs = x
        x = x.permute((0, 2, 1))
        
        
        x = x.reshape((-1, self.embedding_dim))

        unquantized = x
        
        a2 = torch.sum(x.square(), dim=1, keepdim=True)
        
        ab2 = 2 * torch.matmul(x, self.embeddings)
        b2 = torch.sum(self.embeddings.square(), dim=0, keepdim=True)
        distance = a2 - ab2 + b2

      
        encoding_indices = (-distance).argmax(1)


        # the following commented code describes what you must do
        # if you want to "hard set" vector quantization.
        #
        # we check seq_len to see what part of the vector quantization we are at.
        # e.g. if we are in a 256x256 image, the high level VQ part has seq_len=1024
        
        '''
        if(seq_len == 64):

            #print(encoding_indices.shape)
            for i in range(64*6):

                #start = 64 * i
                #for k in range(start, start + 32):

                if(i < (64 * 3)):
                    encoding_indices[i] = 0 # random.randrange(0, 16)
                else:
                    encoding_indices[i] = 15 # random.randrange(0, 16)

        if(seq_len == 1024):

            for k in range(seq_len * 6):

                encoding_indices[k] = 3

        '''
      
        one_hot = F.one_hot(encoding_indices, num_classes=self.num_embeddings)
       
        quantized = torch.index_select(self.embeddings.T, 0, encoding_indices)
        
        # dimensions here are [batchsize, dimension, embedding_size]    
        quantized = quantized.reshape((-1, seq_len, self.embedding_dim)).permute(0, 2, 1)
        
        e_latent_loss = (quantized - inputs.detach()).square().mean()
        q_latent_loss = (quantized.detach() - inputs).square().mean()
        loss = q_latent_loss + e_latent_loss

        
        quantized = inputs + (quantized - inputs).detach()

        # dimensions here are [batchsize, embedding_size]
        indices_output = encoding_indices.reshape((-1, seq_len))

        
            
        return self.Output(
            quantized, 
            loss, 
            one_hot.reshape(bs, seq_len, -1).permute(0, 2, 1), 
            indices_output, 
            distance.reshape(bs, seq_len, -1).permute(0, 2, 1),
            unquantized,
            self.embeddings.T
        )


class VectorQuantizer2D(VectorQuantizer1D):
    def __init__(self, embedding_dim, num_embeddings):
        super().__init__(embedding_dim, num_embeddings)

    def forward(self, x) -> VectorQuantizer1D.Output:
        bs, c, w, h = x.shape
        x = x.reshape(bs, c, w * h)
        x = super().forward(x)

        indices_output = x.encoding_indices.reshape((-1, w, h))
     
        return self.Output(
            x.quantized.reshape(bs, c, w, h), 
            x.loss, 
            x.one_hot.reshape(bs, -1, w, h), 
            indices_output, 
            x.distances.reshape(bs, -1, w, h),
            x.unquantized,
            x.embeddings
        )

    def quantize_indices(self, indices):
        
        quantized = super().quantize_indices(indices)
        quantized = quantized.reshape((-1, indices.shape[1], indices.shape[2], self.embedding_dim)).permute(0, 3, 1, 2)
        return quantized
