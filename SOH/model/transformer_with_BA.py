import torch
import torch.nn as nn
from typing import Tuple
from torch import Tensor
from torch.nn.modules.linear import NonDynamicallyQuantizableLinear
from torch.nn.init import constant_, zeros_, xavier_normal_, normal_
from torch.nn.parameter import Parameter
from typing import Optional, Union, Callable
from torch.nn import functional as F
from torch.nn.modules.module import Module
from torch.nn.init import xavier_uniform_, zeros_, normal_
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.transformer import _get_activation_fn, _get_clones
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import math
import warnings
from utils import to_var
from model.tasks import to_array, to_tensor
import numpy as np

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class MultiheadAttention(Module):
    
    __constants__ = ['batch_first']
    bias_k: Optional[torch.Tensor]
    bias_v: Optional[torch.Tensor]

    def __init__(self, embed_dim, num_heads, dropout=0., bias=False, add_bias_kv=False, add_zero_attn=False, battery_info_rank=0,
                 kdim=None, vdim=None, battery_num=None, batch_first=False, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(MultiheadAttention, self).__init__()
        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self.cdim = battery_num if battery_num is not None else 1
        self._qkv_same_embed_dim = False

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj_weight = Parameter(torch.empty((embed_dim, embed_dim), **factory_kwargs))
        self.k_proj_weight = Parameter(torch.empty((embed_dim, self.kdim), **factory_kwargs))
        self.v_proj_weight = Parameter(torch.empty((embed_dim, self.vdim), **factory_kwargs))
        self.battery_info_rank = battery_info_rank
        self.battery_scale = 1/battery_info_rank if battery_info_rank > 0 else 0
        self.qlc = Parameter(torch.empty((embed_dim, battery_info_rank, self.cdim), **factory_kwargs))
        self.qrc = Parameter(torch.empty((embed_dim, battery_info_rank, self.cdim), **factory_kwargs))
        self.klc = Parameter(torch.empty((embed_dim, battery_info_rank, self.cdim), **factory_kwargs))
        self.krc = Parameter(torch.empty((self.kdim, battery_info_rank, self.cdim), **factory_kwargs))
        self.vlc = Parameter(torch.empty((embed_dim, battery_info_rank, self.cdim), **factory_kwargs))
        self.vrc = Parameter(torch.empty((self.vdim, battery_info_rank, self.cdim), **factory_kwargs))
        self.olc = Parameter(torch.empty((embed_dim, battery_info_rank, self.cdim), **factory_kwargs))
        self.orc = Parameter(torch.empty((embed_dim, battery_info_rank, self.cdim), **factory_kwargs))
        self.register_parameter('in_proj_weight', None)

        if bias:
            self.in_proj_bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs))
        else:
            self.register_parameter('in_proj_bias', None)
        self.out_proj = NonDynamicallyQuantizableLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

        if add_bias_kv:
            self.bias_k = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
            self.bias_v = Parameter(torch.empty((1, 1, embed_dim), **factory_kwargs))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self._reset_parameters()

    def _reset_parameters(self):
        if self._qkv_same_embed_dim:
            xavier_uniform_(self.in_proj_weight)
        else:
            xavier_uniform_(self.q_proj_weight)
            xavier_uniform_(self.k_proj_weight)
            xavier_uniform_(self.v_proj_weight)
            
        zeros_(self.qlc)
        zeros_(self.klc)
        zeros_(self.vlc)
        zeros_(self.olc)
        normal_(self.qrc)
        normal_(self.krc)
        normal_(self.vrc)
        normal_(self.orc)

        if self.in_proj_bias is not None:
            constant_(self.in_proj_bias, 0.)
            constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            xavier_normal_(self.bias_v)

    def __setstate__(self, state):
        # Support loading old MultiheadAttention checkpoints generated by v1.1.0
        if '_qkv_same_embed_dim' not in state:
            state['_qkv_same_embed_dim'] = True

        super(MultiheadAttention, self).__setstate__(state)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, car: Tensor = None, key_padding_mask: Optional[Tensor] = None,
                need_weights: bool = True, attn_mask: Optional[Tensor] = None,
                average_attn_weights: bool = True) -> Tuple[Tensor, Optional[Tensor]]:
        
        is_batched = query.dim() == 3
        if self.batch_first and is_batched:
            # make sure that the transpose op does not affect the "is" property
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]
        
        car = car.unsqueeze(1)
        battery_info_q = torch.matmul(torch.matmul(self.qlc, car).squeeze(2), torch.matmul(self.qrc,car).squeeze(2).T)
        battery_info_k = torch.matmul(torch.matmul(self.klc, car).squeeze(2), torch.matmul(self.krc,car).squeeze(2).T)
        battery_info_v = torch.matmul(torch.matmul(self.vlc, car).squeeze(2), torch.matmul(self.vrc,car).squeeze(2).T)
        battery_info_o = torch.matmul(torch.matmul(self.olc, car).squeeze(2), torch.matmul(self.orc,car).squeeze(2).T)
        #print(battery_info.shape)
        
        q = self.q_proj_weight + battery_info_q * self.battery_scale
        k = self.k_proj_weight + battery_info_k * self.battery_scale
        v = self.v_proj_weight + battery_info_v * self.battery_scale
        o = self.out_proj.weight + battery_info_o * self.battery_scale
        #print(battery_info, v)
        
        attn_output, attn_output_weights = F.multi_head_attention_forward(
                query, key, value, self.embed_dim, self.num_heads,
                self.in_proj_weight, self.in_proj_bias,
                self.bias_k, self.bias_v, self.add_zero_attn,
                self.dropout, o, self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask, use_separate_proj_weight=True,
                q_proj_weight=q, k_proj_weight=k,
                v_proj_weight=v, average_attn_weights=average_attn_weights)
        
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), attn_output_weights
        else:
            return attn_output, attn_output_weights

class TransformerEncoderLayer(Module):
    
    __constants__ = ['batch_first', 'norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, battery_num: int = 1, battery_info_rank: int = 1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, battery_num=battery_num, battery_info_rank=battery_info_rank,
                                            **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            activation = _get_activation_fn(activation)

        self.activation = activation

    def __setstate__(self, state):
        super(TransformerEncoderLayer, self).__setstate__(state)
        if not hasattr(self, 'activation'):
            self.activation = F.relu


    def forward(self, src: Tensor, car: Tensor = None) -> Tensor:
        
        x = src
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), car)
            x = x + self._ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self._sa_block(x, car))
            x = self.norm2(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, car: Tensor) -> Tensor:
        x = self.self_attn(x, x, x, car=car,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout2(x)

class TransformerEncoder(Module):
    
    def __init__(self, encoder_layers, num_layers):
        super(TransformerEncoder, self).__init__()
        self.layers = ModuleList(encoder_layers)
        self.num_layers = num_layers

    def forward(self, src: Tensor, car: Tensor) -> Tensor:
        
        output = src

        for mod in self.layers:
            output = mod(output, car=car)

        return output

class TransformerDecoderLayer(Module):
    
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1, battery_num: int = 1, battery_info_rank: int = 1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, battery_num=battery_num, battery_info_rank=battery_info_rank,
                                            bias=bias, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first, battery_num=battery_num, battery_info_rank=battery_info_rank,
                                                 bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm2 = RMSNorm(d_model, eps=layer_norm_eps)
        self.norm3 = RMSNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        car: Tensor
    ) -> Tensor:
        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), car=car)
            x = x + self._mha_block(self.norm2(x), memory, car=car)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, car=car))
            x = self.norm2(x + self._mha_block(x, memory, car=car))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor, car: Tensor = None) -> Tensor:
        x = self.self_attn(x, x, x, car=car,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor, car: Tensor = None) -> Tensor:
        x = self.multihead_attn(x, mem, mem, car=car,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)

class TransformerDecoder(Module):
        
    def __init__(self, decoder_layers, num_layers):
        super(TransformerDecoder, self).__init__()
        self.layers = ModuleList(decoder_layers)
        self.num_layers = num_layers
    
    def forward(self, tgt: Tensor, memory: Tensor, car: Tensor) -> Tensor:
        output = tgt

        for mod in self.layers:
            output = mod(output, memory, car = car)

        return output

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=1000):
        super(PositionalEmbedding, self).__init__()
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, dtype=torch.float)
        pe.require_grad = False
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        self.check_nan = False

    def forward(self, x):
        return self.pe[:, :x.size(1)]

class DynamicVAE(nn.Module):

    def __init__(self, nhead, dim_feedforward, kernel_size, hidden_size, latent_size, seq_length, battery_num, encoder_embedding_size, output_embedding_size,
                 decoder_embedding_size, battery_info_rank = [0], num_layers=1, decoder_low_rank=False, **params):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.decoder_low_rank = decoder_low_rank
        battery_info_rank += [0]*num_layers
        print(battery_info_rank)
        encoder_layers = []
        self.position_embedding = PositionalEmbedding(d_model=hidden_size)
        for i in range(num_layers):
            encoder_layers.append(TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, battery_num=battery_num, battery_info_rank = battery_info_rank[i],
                                                   dim_feedforward=dim_feedforward, activation = F.silu, norm_first = True, batch_first=True))
        self.encoder_rnn = TransformerEncoder(
            encoder_layers, num_layers=num_layers)
        if decoder_low_rank:
            decoder_layers = []
            for i in range(num_layers):
                decoder_layers.append(TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, battery_num=battery_num, battery_info_rank = battery_info_rank[num_layers - i - 1],
                                                    dim_feedforward=dim_feedforward, activation = F.silu, norm_first = True, batch_first=True))
            self.decoder_rnn = TransformerDecoder(
                decoder_layers, num_layers=num_layers)
        else:
            self.decoder_rnn = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, activation = F.silu, norm_first = True, batch_first=True), num_layers=num_layers)
        self.sequence2encoder = nn.Conv1d(encoder_embedding_size,
                                          hidden_size,
                                          kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.sequence2decoder = nn.Conv1d(decoder_embedding_size,
                                          hidden_size,
                                          kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        
        self.hiddenconv = nn.Conv1d(in_channels=seq_length, out_channels=1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        self.hidden2mean = nn.Linear(hidden_size, latent_size)
        self.hidden2log_v = nn.Linear(hidden_size, latent_size)
        self.latent2hidden = nn.Linear(latent_size, hidden_size)
        self.convhidden = nn.ConvTranspose1d(in_channels=1, out_channels=seq_length, kernel_size=kernel_size, padding=(kernel_size - 1) // 2)
        
        self.outputs2embedding = nn.Linear(hidden_size, output_embedding_size)
        
        self.mean2latent = nn.Sequential(nn.Linear(latent_size, int(hidden_size / 2)), nn.ReLU(),
                                         nn.Linear(int(hidden_size / 2), 1))

    def forward(self, input_sequence, encoder_filter, decoder_filter, seq_lengths, car, cn, noise_scale=1.0):
        batch_size = input_sequence.size(0)
        en_input_sequence = encoder_filter(input_sequence)
        en_input_sequence = en_input_sequence.to(torch.float32)
        # 形状为（batch_size,seq_len,encoder_embedding_size)
        # 调换后俩个维，形状为（batch_size,encoder_embedding_size,seq_len)
        en_input_sequence = en_input_sequence.permute(0, 2, 1)
        # 形状为（batch_size,hidden_size,seq_len)
        en_input_embedding = self.sequence2encoder(en_input_sequence)
        en_input_embedding = en_input_embedding.permute(
            0, 2, 1)  # 形状为（batch_size,seq_len,hidden_size)
        # 形状为（batch_size,hidden_size)
        en_input_embedding = en_input_embedding + self.position_embedding(en_input_embedding)
        car = to_var(car).to(torch.float32)
        en_output = self.encoder_rnn(en_input_embedding, car=car)
        
        hidden = self.hiddenconv(en_output).squeeze(1) # (B, H)
        
        mean = self.hidden2mean(hidden) # (B, L)
        log_v = self.hidden2log_v(hidden) # (B, L)
        std = torch.exp(0.5 * log_v) # (B, L)
        mean_pred = self.mean2latent(mean)  # (B, 1)
        
        z = to_var(torch.randn(mean.shape))
        if self.training:
            z = z * std * noise_scale + mean
        else:
            z = mean
        
        hidden = self.latent2hidden(z).unsqueeze(1)# (B, 1, H)
        hidden = self.convhidden(hidden)
        
        de_input = hidden  # (B, S, H)
        de_input_sequence = decoder_filter(input_sequence)
        de_input_sequence = de_input_sequence.to(torch.float32)
        de_input_sequence = de_input_sequence.permute(0, 2, 1)
        de_input_embedding = self.sequence2decoder(de_input_sequence)
        de_input_embedding = de_input_embedding.permute(0, 2, 1)
        de_input_embedding = de_input_embedding + self.position_embedding(de_input_embedding)             
        if self.decoder_low_rank:
            outputs = self.decoder_rnn(de_input_embedding, de_input, car=car)
        else:
            outputs = self.decoder_rnn(de_input_embedding, de_input)
        # （batch_size,seqlen,output_embedding_size)
        log_p = self.outputs2embedding(outputs)
        return log_p, mean, log_v, z, mean_pred

