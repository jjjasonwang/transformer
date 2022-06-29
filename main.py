# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.



import torch
import torch.nn as nn
import numpy as np
import math
import torch.nn.functional as F

class Config(object):
    def __init__(self):
        self.vocab_size = 6

        self.d_model = 20
        self.n_heads = 2

        self.dim_k  = self.d_model / self.n_heads
        self.dim_v = self.d_model / self.n_heads

        self.padding_size = 30
        self.UNK = 5
        self.PAD = 4

        self.N = 6
        self.p = 0.1

config = Config()

# 1. Embedding layer
class Embedding(nn.Module):
    def __init__(self, vocab_size):
        super(Embedding, self).__init__()
        #num_embeddings, embedding_dim, padding_idx
        self.embedding = nn.Embedding(vocab_size, config.d_model, padding_idx=config.PAD)

    def forward(self, x):
        #根据句子长度进行padding. 截长补短
        # x: batch_size * seq_len
        for i in range(len(x)):
            if len(x[i]) < config.padding_size:
                x[i].append([config.UNK] * (config.padding_size - len(x[i])))
            else:
                x[i] = x[i][:config.padding_size]
        x = self.embedding(torch.tensor(x))
        return x

# 2.Positional Encoding layer
class Positional_Encoding(nn.Module):
    def __init__(self, d_model):
        super(Positional_Encoding, self).__init__()
        self.d_model = d_model

    def forward(self, seq_len, embedding_dim):
        positional_encoding = np.zeros((seq_len, embedding_dim))
        for pos in range(positional_encoding[0]):
            for i in range(positional_encoding[1]):
                positional_encoding[pos][i] = math.sin(math.sin(pos/(10000**(2*i/self.d_model))) if i % 2 == 0 else math.cos(pos/(10000**(2*i/self.d_model))))
        return torch.from_numpy(positional_encoding)

# 3.Encoder
# 3.1 Multihead Attention
class Multihead_Attention(nn.Module):
    def __init__(self, d_model, dim_k, dim_v, n_heads):

        super(Multihead_Attention, self).__init__()
        self.dim_v = dim_v
        self.dim_k = dim_k
        self.n_heads = n_heads

        self.q = nn.Linear(d_model, dim_k)
        self.k = nn.Linear(d_model, dim_k)
        self.v = nn.Linear(d_model, dim_v)
        self.o = nn.Linear(dim_v, d_model)
        self.norm_fact = 1 / math.sqrt(d_model)

    #mask中的长度为seq_len
    def generate_mask(self, dim):
        matrix = np.zeros((dim, dim))
        mask = torch.tensor(np.tril(matrix))
        return mask == 1

    # x: batch_size * seq_len * d_model(embedding_size)
    def forward(self,x,y,requires_mask=False):
        Q = self.q(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        K = self.k(x).reshape(-1, x.shape[0], x.shape[1], self.dim_k // self.n_heads)
        V = self.v(y).reshape(-1, x.shape[0], x.shape[1], self.dim_v // self.n_heads)

        attention_score = torch.matmul(Q,K.permutate(0,1,3,2)) * self.norm_fact

        if requires_mask:
            mask = self.generate_mask(x.shape[1])
            attention_score.masked_fill(mask, value=float("-inf"))
        attention_score = F.softmax(attention_score)
        output = torch.matmul(attention_score, V).reshape(y.shape[0], y.shape[1], -1)
        output = self.o(output)
        return output

# 3.2 FFN
class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim=2048):
        super(FFN,self).__init__()
        super.L1 = nn.Linear(input_dim, hidden_dim)
        super.L2 = nn.Linear(hidden_dim, input_dim)

    def forward(self,x):
        output = F.relu(self.L1(x))
        output = self.L2(output)
        return output

# 3.3 Add & Norm
class Add_Norm(nn.Module):
    def __init__(self):
        self.dropout = nn.Dropout(config.p)
        super(Add_Norm, self).__init__()

    def forward(self, x, sub_layer, **kwargs):
        sub_out = sub_layer(x, **kwargs)
        x = self.dropout(x + sub_out)
        layer_norm = nn.LayerNorm(x.size()[1:])
        out = layer_norm(x)
        return out

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.multihead_attention = Multihead_Attention(config.d_model, config.dim_k, config.dim_v, config.n_heads)
        self.add_norm = Add_Norm()
        self.ffn = FFN(config.d_model)

    def forward(self, x):
        x += self.positional_encoding(x.shape[1], config.d_model)
        output = self.add_norm(x, self.multihead_attention, y=x)
        output = self.add_norm(output, self.ffn)

        return output


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.positional_encoding = Positional_Encoding(config.d_model)
        self.multihead_attention = Multihead_Attention(config.d_model, config.dim_k, config.dim_v, config.n_heads)
        self.ffn = FFN(config.d_model)
        self.add_norm = Add_Norm()

    def forward(self, x, encoder_output):
        x += self.positional_encoding(x.shape[1], config.d_model)
        output = self.add_norm(x, self.multihead_attention, y=x, requires_mask=True)
        output = self.add_norm(output, self.multihead_attention, y=encoder_output, requires_mask=True)
        output = self.add_norm(output, self.ffn)
        return output

class Transformer_layer(nn.Module):
    def __init__(self):
        super(Transformer_layer, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        x_input, x_output = x
        encoder_output = self.encoder(x_input)
        decoder_output = self.decoder(x_output, encoder_output)

        return (encoder_output, decoder_output)

class Transformer(nn.Module):
    def __init__(self, N, vocab_size, output_dim):
        super(Transformer, self).__init__()
        self.embedding_input = Embedding(vocab_size=vocab_size)
        self.embedding_output = Embedding(vocab_size=vocab_size)

        self.output_dim = output_dim
        self.linear = nn.Linear(config.d_model, output_dim)
        #output: batchsize * seq_len * d_model(embedding_size)  ->   batchsize * seq_len * output_dim
        self.softmax = nn.Softmax(dim=-1)
        self.model = nn.Sequential(*[Transformer_layer() for _ in range(N)])

    def forward(self, x):
        x_input, x_output = x
        x_input = self.embedding_input(x_input)
        x_output = self.embedding_output(x_output)

        _, output = self.model((x_input, x_output))
        output = self.linear(output)
        output = self.softmax(output)
        return output


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print('a')
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
