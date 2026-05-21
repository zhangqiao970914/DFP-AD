from torch.nn import functional as F
import torch
import torch.nn as nn
import math
from einops import rearrange, repeat
import numpy as np
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

def seq_to_image(x, H=None, W=None):
    B, L, C = x.shape
    if H is None or W is None:
        side = int(math.sqrt(L))
        assert side * side == L, f"Sequence length L={L} is not a perfect square, cannot infer H and W"
        H = W = side
    assert L == H * W, f"Sequence length L={L} must equal H*W={H*W}"
    x = x.permute(0, 2, 1).contiguous()  # [B, C, L]
    x = x.view(B, C, H, W)               # [B, C, H, W]
    return x

def image_to_seq(x):
    B, C, H, W = x.shape
    L = H * W
    x = x.view(B, C, L)                  # [B, C, L]
    x = x.permute(0, 2, 1).contiguous()  # [B, L, C]
    return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class bMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class sMlp(nn.Module):
    def __init__(self, dim, mlp_ratio=4, out_features=None, drop=0.,
                bias=False, **kwargs):
        super().__init__()
        in_features = dim
        out_features = out_features or in_features
        hidden_features = int(mlp_ratio * in_features)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = StarReLU()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.relu = nn.ReLU(inplace=inplace)
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                                  requires_grad=scale_learnable)
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                                 requires_grad=bias_learnable)
    def forward(self, x):
        return self.scale * self.relu(x) ** 2 + self.bias

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class Dynamic_Frequency_Demodulator(nn.Module):
    def __init__(self, in_channels, kernel_size=3, stride=1, groups=8):
        super().__init__()
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.groups = groups
        self.stride = stride
        self.low_ap = nn.AdaptiveAvgPool2d((1, 1))
        self.low_conv = nn.Conv2d(in_channels, groups * kernel_size**2, kernel_size=1, bias=False)
        self.low_gate_conv = nn.Conv2d(groups * kernel_size**2, groups * kernel_size**2, kernel_size=1, bias=False)
        self.low_bn = nn.GroupNorm(num_groups=groups, num_channels=groups * kernel_size**2)
        self.low_act = nn.Softmax(dim=-2)
        self.low_sigmoid = nn.Sigmoid()
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(self, x_seq):
        x = seq_to_image(x_seq)
        identity_input = x 
        B, C, H, W = x.shape

        low_feat = self.low_ap(x)  
        low_filter = self.low_conv(low_feat)
        gate = self.low_sigmoid(self.low_gate_conv(low_filter))
        low_filter = low_filter * (1 + gate)
        low_filter = self.low_bn(low_filter)

        x_unfold = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            B, self.groups, C // self.groups, self.kernel_size**2, H * W)
        _, c1, p, q = low_filter.shape
        low_filter_reshaped = low_filter.reshape(B, c1 // self.kernel_size**2, self.kernel_size**2, p * q).unsqueeze(2)
        low_filter_reshaped = self.low_act(low_filter_reshaped)

        low_part = torch.sum(x_unfold * low_filter_reshaped, dim=3).reshape(B, C, H, W)
        high_part = identity_input - low_part
        return image_to_seq(low_part), image_to_seq(high_part)


class Prototype_Alignment(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.learn_scale = nn.Parameter(torch.ones(num_heads, 1, 1), requires_grad=True)
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim *2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, prototype_token):
        B, N, C = x.shape
        prototype_num = prototype_token.shape[1]
        q = self.q(x).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(prototype_token).reshape(B, prototype_num, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.learn_scale
        attn = F.relu(attn)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Prototype_Learning(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, y):
        B, T, C = x.shape
        _, N, _ = y.shape
        q = self.q(x).reshape(B, T, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)[0]
        kv = self.kv(y).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attnmap = attn.softmax(dim=-1)
        attn = self.attn_drop(attnmap)
        x = (attn @ v).transpose(1, 2).reshape(B, T, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Prototype_Learning_Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Prototype_Learning(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x, y):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(y)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class Dynamic_Frquency_Modulator(nn.Module):
    def __init__(self, dim, expansion_ratio=1, reweight_expansion_ratio=.0625, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=14, weight_resize=True, group=32, init_scale=1e-5,
                 **kwargs):
        super().__init__()
        self.size = size
        self.filter_size = size // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.reweight = sMlp(dim, reweight_expansion_ratio, group * num_filters, bias=False)
        self.complex_weights = nn.Parameter(
            torch.randn(num_filters, dim//group, self.size, self.filter_size,dtype=torch.float32) * init_scale)
        trunc_normal_(self.complex_weights, std=init_scale)
        self.act2 = act2_layer()

    def forward(self, x):
        B, C, H, W, = x.shape
        x_rfft = torch.fft.rfft2(x.to(torch.float32), dim=(2, 3), norm='ortho')
        B, C, RH, RW, = x_rfft.shape
        x = x.permute(0, 2, 3, 1)
        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, -1, self.num_filters).tanh_() 
        weight = self.complex_weights
        if not weight.shape[2:4] == x_rfft.shape[2:4]:
            weight = F.interpolate(weight, size=x_rfft.shape[2:4], mode='bicubic', align_corners=True)
        weight = torch.einsum('bgf,fchw->bgchw', routeing, weight)
        weight = weight.reshape(B, C, RH, RW)
        x_rfft = torch.view_as_complex(torch.stack([x_rfft.real * weight, x_rfft.imag * weight], dim=-1))
        x = torch.fft.irfft2(x_rfft, s=(H, W), dim=(2, 3), norm='ortho')
        return x

class Prototype_Aligned_Frequency_Modulator(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim//2)
        self.attn = Prototype_Alignment(dim//2, num_heads=num_heads//2, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.proj = nn.Linear(dim, dim//2)
        self.proj_low = nn.Linear(dim, dim//2)
        self.proj_high = nn.Linear(dim, dim//2)
        self.conv = nn.Conv2d(dim, dim, kernel_size=3, padding=1, stride=1, groups=dim)
        
        self.DFD = Dynamic_Frequency_Demodulator(dim//2)
        self.DFM = Dynamic_Frquency_Modulator(dim=dim, expansion_ratio=1, reweight_expansion_ratio=.0625, group=16, num_filters=4, size=28, act2_layer=nn.Identity, bias=False, weight_resize=True)

    def forward(self, x, low_prototype, high_prototype, return_attention=False):
        x = self.proj(x)
        low = self.proj_low(low_prototype)
        high = self.proj_high(high_prototype)

        # Dynamic Frequency Demodulator
        y_low, y_high = self.DFD(x) 

        # LF Prototype Alignment
        y_low, attn_low = self.attn(self.norm1(y_low), low)   
        # HF Prototype Alignment
        y_high, attn_high = self.attn(self.norm1(y_high), high) 

        # Dynamic Frequency Demodulator
        y = torch.cat([y_low, y_high], dim=2) 
        y = self.conv(seq_to_image(y))
        y = self.DFM(y) + y
        y = image_to_seq(y)
        attn = attn_low + attn_high
        x = self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


