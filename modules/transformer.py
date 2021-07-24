import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F
from torch import nn, Tensor
from typing import Optional, List

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation=nn.ReLU(), normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if tensor.shape[0] < pos.shape[0]: pos = pos[:tensor.shape[0]]
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation= nn.ReLU(), normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if tensor.shape[0] < pos.shape[0]: pos = pos[:tensor.shape[0]]
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)


class PositionEmbedding2DLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos



class TransformerEncoder(nn.Module):
    def __init__(self, enc_seqlen, d_model=512, nhead=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU()):
        super().__init__()
        self.encoder_layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, 
            dropout, activation, False) for _ in range(num_encoder_layers)])
        self.enc_embed = nn.Embedding(enc_seqlen, d_model)
        
    def forward(self, src, mask):
        """
        Shape = (bs, d, seq)
        """
        bs = src.shape[0]
        src = src.permute(2, 0, 1)
        m = src 
        enc_embed = self.enc_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        for layer in self.encoder_layers:
            m = layer(m,
                pos=enc_embed,
                src_mask = mask
            )
        return m.permute(1, 2, 0), enc_embed.permute(1, 2, 0)

class TransformerDecoder(nn.Module):
    def __init__(self, dec_seqlen, d_model=512, nhead=8, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU()):
        super().__init__()
        self.decoder_layers = nn.ModuleList([TransformerDecoderLayer(d_model, nhead, dim_feedforward,
            dropout, activation, False) for _ in range(num_decoder_layers)])
        self.decoder_norm = nn.LayerNorm(d_model)
        self.dec_embed = nn.Embedding(dec_seqlen, d_model)

    def forward(self, tgt, m, enc_embed, mask):
        """
        Shape = (bs, d, seq)
        """
        bs = tgt.shape[0]
        enc_embed = enc_embed.permute(2, 0, 1)
        m = m.permute(2, 0, 1)
        tgt = tgt.permute(2, 0, 1)
        dec_embed = self.dec_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        out = tgt
        for layer in self.decoder_layers:
            out = layer(out, m, 
                pos=enc_embed,
                query_pos=dec_embed
            )
  
        return self.decoder_norm(out).permute(1, 2, 0), dec_embed.permute(1, 2, 0)

class Transformer(nn.Module):
    def __init__(self, enc_seqlen, dec_seqlen, d_model=512, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation=nn.LeakyReLU()):
        super().__init__()
        self.encoder = TransformerEncoder(
            enc_seqlen, 
            dec_seqlen, 
            d_model=d_model, 
            nhead=nhead, 
            num_encoder_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation)
        self.decoder = TransformerDecoder(
            enc_seqlen, 
            dec_seqlen, 
            d_model=d_model, 
            nhead=nhead, 
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward, dropout=dropout,
            activation=activation)
        
    def forward(self, src, tgt, enc_mask=None, dec_mask=None):
        """
        Shape = (bs, d, seq)
        """
        m, enc_embed = self.encoder(src, enc_mask)
        # mask = mask.flatten(1)
        out, dec_embed = self.decoder(tgt, m, enc_embed, dec_mask)   
        return out, m, enc_embed, dec_embed


class DiscriminativeTransformer(nn.Module):
    def __init__(self,
            d_model: int,
            enc_seqlen: int,
            proj_dim: int,
            num_querys: int=1,
            num_decoder_layers: int=6,
            num_encoder_layers: int=0,
            **kwargs
        ) -> None:
        '''
            Args:
                blocks
                enc_seqlen
                proj_dim
                num_querys
                num_layers

            Keyword Args:
                nhead,
                dim_feedforward,
                dropout,
                activation
        '''
        super().__init__()
        self.num_querys = num_querys
        self.transformer_dim = d_model
        self.performer_encoder = TransformerEncoder(
            d_model=self.transformer_dim,
            enc_seqlen=enc_seqlen, 
            num_encoder_layers=num_encoder_layers, 
            **kwargs)
        self.performer_decoder = TransformerDecoder(
            d_model=self.transformer_dim,
            dec_seqlen=num_querys,
            num_decoder_layers=num_decoder_layers,
            **kwargs)
        self.proj = nn.Linear(self.transformer_dim, proj_dim)


    def forward(self, x):
        q = torch.zeros(self.transformer_dim, self.num_querys, device=x.device)
        query = q[None, ...].repeat(x.shape[0], 1, 1)
        x = x.flatten(2)
        _, enc_embed = self.performer_encoder(x, None)
        x, _ = self.performer_decoder(query, x, enc_embed, None)
        x = x.permute(0, 2, 1)
        x = self.proj(x)
        return x
