import torch.nn as nn
import torch
from torch import Tensor
from einops.layers.torch import Rearrange
from einops import rearrange
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import math



def flip(x, dim):
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous()
    x = x.view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:, getattr(torch.arange(x.size(1)-1,
                      -1, -1), ('cpu','cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)

def sinc(band,t_right):
    y_right= torch.sin(2*math.pi*band*t_right)/(2*math.pi*band*t_right)
    y_left= flip(y_right,0)

    y=torch.cat([y_left,Variable(torch.ones(1)).cuda(),y_right])

    return y


class sinc_conv(nn.Module):
    def __init__(self, N_filt, Filt_dim, fs):
        super(sinc_conv, self).__init__()
        low_freq_mel = 4
        high_freq_mel = 38

        np.random.seed(42)
        b1 = np.random.uniform(low_freq_mel,high_freq_mel, N_filt)
        b2 = np.zeros_like(b1) + 2

        self.freq_scale = fs * 1.0
        self.filt_b1 = nn.Parameter(torch.from_numpy(b1 / self.freq_scale))
        self.filt_band = nn.Parameter(torch.from_numpy(b2 / self.freq_scale))

        self.N_filt = N_filt
        self.Filt_dim = Filt_dim
        self.fs = fs

    def forward(self, x):
        filters = Variable(torch.zeros((self.N_filt, self.Filt_dim))).cuda()
        N = self.Filt_dim
        t_right = Variable(torch.linspace(1, (N - 1) / 2, steps=int((N - 1) / 2)) / self.fs).cuda()

        min_freq = 1.0
        min_band = 2.0

        filt_beg_freq = torch.abs(self.filt_b1) +  min_freq / self.freq_scale
        filt_end_freq = torch.clamp(filt_beg_freq + (torch.abs(self.filt_band) + min_band / self.freq_scale), 3/self.freq_scale, (self.fs/2)/self.freq_scale)
        n = torch.linspace(0, N, steps=N)

        window = 0.54 - 0.46 * torch.cos(2 * math.pi * n / N)
        window = Variable(window.float().cuda())

        for i in range(self.N_filt):
            low_pass1 = 2 * filt_beg_freq[i].float() * sinc(filt_beg_freq[i].float() * self.freq_scale, t_right)
            low_pass2 = 2 * filt_end_freq[i].float() * sinc(filt_end_freq[i].float() * self.freq_scale, t_right)
            band_pass = (low_pass2 - low_pass1)

            band_pass = band_pass / torch.max(band_pass)   #normalize to one

            filters[i, :] = band_pass.cuda() * window
        x = x.cuda()
        out = F.conv2d(x, filters.view(self.N_filt, 1, 1,self.Filt_dim),padding=(0, int((self.Filt_dim - 1) / 2)))
        return out


class VarMaxPool1D(nn.Module):
    def __init__(self, T, kernel_size, stride=None, padding=0):
        super().__init__()

        self.kernel_size = kernel_size

        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = stride

        self.padding = padding

    def forward(self, x):
        mean_of_squares = F.avg_pool1d(
            x**2, self.kernel_size, self.stride, self.padding
        )
        # Compute the square of the mean (E[x])^2
        square_of_mean = (
            F.avg_pool1d(x, self.kernel_size, self.stride, self.padding) ** 2
        )

        # Compute the variance: Var[X] = E[X^2] - (E[X])^2
        variance = mean_of_squares - square_of_mean
        # out = self.time_agg(variance)
        out = F.avg_pool1d(variance, variance.shape[-1])

        return out
    
class SSA(nn.Module):
    def __init__(self, T, num_channels, epsilon=1e-5, mode="var"):
        super().__init__()

        self.alpha = nn.Parameter(torch.ones(1, num_channels, 1))
        self.gamma = nn.Parameter(torch.zeros(1, num_channels, 1))
        self.beta = nn.Parameter(torch.zeros(1, num_channels, 1))

        self.epsilon = epsilon
        self.mode = mode

        self.GP = VarMaxPool1D(T, 250)

    def forward(self, x):
        x = x.squeeze(1) 
        B, C, T = x.shape

        if self.mode == "var":
            embedding = (self.GP(x) + self.epsilon).pow(0.5) * self.alpha
            # 通道归一化
            norm = (self.gamma) / (
                embedding.pow(2).mean(dim=1, keepdim=True) + self.epsilon
            ).pow(0.5)

        gate = 1 + torch.tanh(embedding * norm + self.beta)
        out_s = x * gate

        return out_s.unsqueeze(1)


class PatchEmbeddingCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 125), stride=(1, 1), padding=(0, 125//2), bias = False),
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 63), stride=(1, 1), padding=(0, 63//2), bias = False),
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(1, 31), stride=(1, 1), padding=(0, 31//2), bias=False),
        )

        self.S_Conv = nn.Sequential(
            nn.BatchNorm2d(24),
            nn.Conv2d(in_channels=24, out_channels=48, kernel_size=(22, 1), stride=(1, 1), padding='valid', groups=24, bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 5), stride=5),
            nn.Dropout(0.5),

            nn.Conv2d(48, 48, (1, 16), padding=(0, 16//2), groups=48, bias=False), 
            nn.Conv2d(48, 48, (1, 1), bias=False),
            nn.BatchNorm2d(48),
            nn.ELU(),
            nn.AvgPool2d((1, 8), stride=8),
            nn.Dropout(0.5),  
        )
        
    def forward(self, x: Tensor) -> Tensor:
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        x = torch.cat([x1, x2, x3], dim=1)
        x = self.S_Conv(x)

        return x



class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1000):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        
        t = torch.arange(max_seq_len).float()
        freqs = torch.einsum('i,j->ij', t, inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def forward(self, x, seq_len=None):
        if seq_len > self.cos_cached.shape[2]:
             pass 
        return self.cos_cached[:, :, :seq_len, ...], self.sin_cached[:, :, :seq_len, ...]



def rotate_half(x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)



def apply_rotary_pos_emb(x, cos, sin):
    return (x * cos) + (rotate_half(x) * sin)



class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output



class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_size, dropout):
        super().__init__()
        self.emb_size = emb_size
        self.num_heads = num_heads
        
        self.keys = nn.Linear(emb_size, emb_size)
        self.queries = nn.Linear(emb_size, emb_size)
        self.values = nn.Linear(emb_size, emb_size)

        self.att_drop = nn.Dropout(dropout)

        self.projection = nn.Linear(emb_size, emb_size)

        self.rope = RotaryEmbedding(6, max_seq_len=25)

    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        B, N, C = x.shape

        queries = rearrange(self.queries(x), "b n (h d) -> b h n d", h=self.num_heads)
        keys = rearrange(self.keys(x), "b n (h d) -> b h n d", h=self.num_heads)
        values = rearrange(self.values(x), "b n (h d) -> b h n d", h=self.num_heads)

        cos, sin = self.rope(queries, seq_len=N)
        queries = apply_rotary_pos_emb(queries, cos, sin)
        keys = apply_rotary_pos_emb(keys, cos, sin)

        energy = torch.einsum('bhqd, bhkd -> bhqk', queries, keys)

        if mask is not None:
            fill_value = torch.finfo(torch.float32).min
            energy.mask_fill(~mask, fill_value)
        
        scaling = self.emb_size // self.num_heads
        att = F.softmax(energy / math.sqrt(scaling), dim=-1)
        att = self.att_drop(att)
        out = torch.einsum('bhal, bhlv -> bhav ', att, values)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.projection(out)
        return out
    

# PointWise FFN
class FeedForwardBlock(nn.Sequential):
    def __init__(self, emb_size, expansion, fc_drop):
        super().__init__(
            nn.Linear(emb_size, expansion * emb_size),
            nn.GELU(),
            nn.Dropout(fc_drop),
            nn.Linear(expansion * emb_size, emb_size),
        )


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, emb_size, expansion=4, attn_drop=0.5, fc_drop=0.5, drop_path_rate=0.0):
        super().__init__()
        self.multihead_attention = MultiHeadAttention(num_heads, emb_size, attn_drop)
        self.layernorm1 = nn.LayerNorm(emb_size)

        self.feed_forward = FeedForwardBlock(emb_size, expansion, fc_drop)
        self.layernorm2 = nn.LayerNorm(emb_size)

        self.drop_path1 = DropPath(drop_path_rate)
        self.drop_path2 = DropPath(drop_path_rate)

    def forward(self, x):
        attn_out = self.multihead_attention(x)
        attn_out = self.drop_path1(attn_out)  # ←←← Apply DropPath here
        x = x + attn_out
        x = self.layernorm1(x)

        # FFN block with residual + DropPath
        ffn_out = self.feed_forward(x)
        ffn_out = self.drop_path2(ffn_out)    # ←←← Apply DropPath here
        x = x + ffn_out
        x = self.layernorm2(x)
        return x


class TransformerModule(nn.Module):
    def __init__(self, num_heads, depth, emb_size, drop_path_max=0.25):
        super().__init__()

        drop_rates = (torch.linspace(0, 1, depth) ** 2 * drop_path_max).tolist()

        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(
                num_heads=num_heads,
                emb_size=emb_size,
                drop_path_rate=drop_rates[i]
            ) for i in range(depth)
        ])
        
    def forward(self, x):
        for encoder in self.transformer_encoders:
            x = encoder(x)
        return x


class ClassificationHead(nn.Sequential):
    def __init__(self, flatten_number, n_classes):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(1200, n_classes)
        )

    def forward(self, x):
        out = self.fc(x)
        return out


class Mymodel(nn.Module):
    def __init__(self, num_heads=8, 
                 emb_size=48,
                 depth=5, 
                 database_type='A', 
                 flatten_eeg1 = 600,
                ):
        super().__init__()
        self.number_class = 4
        self.number_channel = 22
        self.emb_size = emb_size
        self.flatten_eeg1 = flatten_eeg1

        self.sinc_conv = sinc_conv(N_filt=18, Filt_dim = 31, fs=250)
        self.sinc_fuse = nn.Conv2d(in_channels=18, out_channels=1, kernel_size=(1, 1), bias=False)

        self.ssa = SSA(1000, 22)

        self.cnn = PatchEmbeddingCNN()
        
        self.projection = nn.Sequential(
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        #self.position = PositioinalEncoding(emb_size, dropout=0.1)
        self.trans = TransformerModule(num_heads, depth, emb_size)

        self.flatten = nn.Flatten()
        self.classification = ClassificationHead(self.flatten_eeg1 , self.number_class) # FLATTEN_EEGNet + FLATTEN_cnn_module


    def forward(self, x):

        x = self.sinc_conv(x)
        x = self.sinc_fuse(x)
        x = self.ssa(x)

        cnn = self.cnn(x)

        cnn = self.projection(cnn)

        # positional embedding
        cnn = cnn * math.sqrt(self.emb_size)
        
        trans = self.trans(cnn)
        # residual connect
        features = cnn+trans
        
        out = self.classification(self.flatten(features))

        # return features, out
        return out