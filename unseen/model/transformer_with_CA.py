import torch
import torch.nn as nn
from torch.nn import functional as F

import math
from utils import to_var
from model.tasks import to_array, to_tensor
import numpy as np

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
        self.position_embedding = PositionalEmbedding(d_model=hidden_size)
        self.battery2embedding = nn.Linear(battery_num, hidden_size)
        self.encoder_rnn =  nn.TransformerDecoder(
                nn.TransformerDecoderLayer(d_model=hidden_size, nhead=nhead, dim_feedforward=dim_feedforward, activation = F.silu, norm_first = True, batch_first=True), num_layers=num_layers)
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
        ba_embedding = self.battery2embedding(car).repeat(batch_size, seq_lengths, 1)
        en_output = self.encoder_rnn(en_input_embedding, ba_embedding)
        
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
        outputs = self.decoder_rnn(de_input_embedding, de_input)
        log_p = self.outputs2embedding(outputs)
        return log_p, mean, log_v, z, mean_pred

