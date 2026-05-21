import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from models.vision_transformer import Dynamic_Frequency_Demodulator


class DFP_AD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            aggregation,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            remove_class_token=False,
            encoder_require_grad_layer=[],
            prototype_token1=None,
            prototype_token2=None,
    ) -> None:
        super(DFP_AD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.aggregation = aggregation
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer
        self.prototype_token1 = prototype_token1[0]
        self.prototype_token2 = prototype_token2[0]
        self.DFD = Dynamic_Frequency_Demodulator(768)
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def gather_loss(self, query, keys):
        self.distribution = 1. - F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        self.distance, self.cluster_index = torch.min(self.distribution, dim=2)
        gather_loss = self.distance.mean()
        return gather_loss

    def soft_coherence_loss(self, query, keys):
        B, N, D = query.size()
        cosine_similarity_matrix = F.cosine_similarity(query.unsqueeze(2), keys.unsqueeze(1), dim=-1)
        norm_weight = F.softmax(cosine_similarity_matrix, dim=-1)
        reweight_keys = torch.bmm(norm_weight, keys)
        self.distribution = 1. - F.cosine_similarity(query.view(B, -1), reweight_keys.view(B, -1), dim=-1)
        soft_coherence_loss = self.distribution.mean()
        return soft_coherence_loss

    def correlation_coherence_loss(self, query, keys, eps=1e-6):

        B, N, C = query.shape
        _, K, _ = keys.shape
    
        q_norm = (query - query.mean(dim=-1, keepdim=True)) / (query.std(dim=-1, keepdim=True) + eps)
        k_norm = (keys - keys.mean(dim=-1, keepdim=True)) / (keys.std(dim=-1, keepdim=True) + eps)
    
        corr = torch.bmm(q_norm, k_norm.transpose(1, 2)) / C
        max_corr, _ = corr.max(dim=-1)   # [B, N]
        corr_loss = 1 - max_corr.mean()
        return corr_loss

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)  
        B, L, _ = x.shape
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)  
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]  

        x = self.fuse_feature(en_list)  
        ### Adaptive Frequency Prototype Extractor (AFPE)

        ### Dynamic Frequency Demodulator ###
        x_low, x_high = self.DFD(x)
        
        ### LF-Prototype Learning ###
        agg_prototype_low = self.prototype_token1  
        for i, blk in enumerate(self.aggregation):
            agg_prototype_low = blk(agg_prototype_low.unsqueeze(0).repeat((B, 1, 1)), x_low)  
        g_loss_low = self.soft_coherence_loss(x_low, agg_prototype_low)  # Low frquency correlation coherence loss

        ### HF-Prototype Learning ###
        agg_prototype_high = self.prototype_token2  
        for i, blk in enumerate(self.aggregation):
            agg_prototype_high = blk(agg_prototype_high.unsqueeze(0).repeat((B, 1, 1)), x_high)  
        g_loss_high = self.soft_coherence_loss(x_high, agg_prototype_high)  # High frquency correlation coherence loss

        g_loss = 0.2 * g_loss_low + 0.2 * g_loss_high

        for i, blk in enumerate(self.bottleneck):  
            x = blk(x)

        de_list = [] # PAFM_Decoder
        for i, blk in enumerate(self.decoder):
            x = blk(x, agg_prototype_low, agg_prototype_high) 
            de_list.append(x)
        de_list = de_list[::-1]  

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]  

        if not self.remove_class_token:  
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de] 

        return en, de, g_loss

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)













































