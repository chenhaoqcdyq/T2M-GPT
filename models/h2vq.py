import torch
import torch.nn as nn
from .encdec import Encoder, Decoder
from .quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset
# from ..builder import SUBMODULES
import torch.nn.functional as F
import numpy as np

class QuantizeEMA_h2vq(nn.Module):
    def __init__(self, nb_code, code_dim, args):
        super().__init__()
        self.nb_code = nb_code
        self.code_dim = code_dim
        self.mu = 0.99
        self.reset_codebook()
        
    def reset_codebook(self):
        self.init = False
        self.code_sum = None
        self.code_count = None
        self.register_buffer('codebook', torch.zeros(self.nb_code, self.code_dim).cuda())

    def _tile(self, x):
        nb_code_x, code_dim = x.shape
        if nb_code_x < self.nb_code:
            n_repeats = (self.nb_code + nb_code_x - 1) // nb_code_x
            std = 0.01 / np.sqrt(code_dim)
            out = x.repeat(n_repeats, 1)
            out = out + torch.randn_like(out) * std
        else :
            out = x
        return out

    def init_codebook(self, x):
        out = self._tile(x)
        self.codebook = out[:self.nb_code]
        self.code_sum = self.codebook.clone()
        self.code_count = torch.ones(self.nb_code, device=self.codebook.device)
        self.init = True
        
    @torch.no_grad()
    def compute_perplexity(self, code_idx) : 
        # Calculate new centres
        code_onehot = torch.zeros(self.nb_code, code_idx.shape[0], device=code_idx.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, code_idx.shape[0]), 1)

        code_count = code_onehot.sum(dim=-1)  # nb_code
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
        return perplexity
    
    @torch.no_grad()
    def update_codebook(self, x, code_idx):
        
        code_onehot = torch.zeros(self.nb_code, x.shape[0], device=x.device)  # nb_code, N * L
        code_onehot.scatter_(0, code_idx.view(1, x.shape[0]), 1)

        code_sum = torch.matmul(code_onehot, x)  # nb_code, w
        code_count = code_onehot.sum(dim=-1)  # nb_code

        # Update centres
        self.code_sum = self.mu * self.code_sum + (1. - self.mu) * code_sum  # w, nb_code
        self.code_count = self.mu * self.code_count + (1. - self.mu) * code_count  # nb_code

        code_update = self.code_sum.view(self.nb_code, self.code_dim) / self.code_count.view(self.nb_code, 1)

        self.codebook = code_update
        prob = code_count / torch.sum(code_count)  
        perplexity = torch.exp(-torch.sum(prob * torch.log(prob + 1e-7)))
            
        return perplexity

    def preprocess(self, x):
        # NCT -> NTC -> [NT, C]
        x = x.permute(0, 2, 1).contiguous()
        x = x.view(-1, x.shape[-1])  
        return x

    def quantize(self, x):
        # Calculate latent code x_l
        k_w = self.codebook.t()
        distance = torch.sum(x ** 2, dim=-1, keepdim=True) - 2 * torch.matmul(x, k_w) + torch.sum(k_w ** 2, dim=0,
                                                                                            keepdim=True)  # (N * L, b)
        _, code_idx = torch.min(distance, dim=-1)
        return code_idx

    def dequantize(self, code_idx):
        x = F.embedding(code_idx, self.codebook)
        return x

    
    def forward(self, x):
        N, width, T = x.shape

        # Preprocess
        x = self.preprocess(x)

        # Init codebook if not inited
        if self.training and not self.init:
            self.init_codebook(x)

        # quantize and dequantize through bottleneck
        code_idx = self.quantize(x)
        x_d = self.dequantize(code_idx)

        # Update embeddings
        if self.training:
            perplexity = self.update_codebook(x, code_idx)
        else : 
            perplexity = self.compute_perplexity(code_idx)
        
        # Loss
        commit_loss = F.mse_loss(x, x_d.detach())

        # Passthrough
        x_d = x + (x_d - x).detach()

        # Postprocess
        x_d = x_d.view(N, T, -1).permute(0, 2, 1).contiguous()   #(N, DIM, T)
        
        return x_d, commit_loss, perplexity, code_idx

# @SUBMODULES.register_module()
class H2VQ_bodyhand(nn.Module):
    def __init__(self,
                 args,
                 nb_code_hand=512,
                 nb_code_body=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                #  face_dim = 150,
                 hand_dim = 90,
                 body_dim = 82,
                 norm=None):
        super().__init__()

        self.code_dim = code_dim
        self.nb_code_hand = nb_code_hand
        self.nb_code_body = nb_code_body
        # self.face_dim = face_dim
        self.hand_dim = hand_dim
        self.body_dim = body_dim
        self.whole_body = self.hand_dim + self.body_dim
        # 初始化手部编码器和量化器
        self.hand_encoder = Encoder(self.hand_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)  # 21个关节*3维
        self.hand_quantizer = QuantizeEMA_h2vq(nb_code_hand, code_dim, args)

        # 初始化身体编码器和量化器
        self.body_encoder = Encoder(self.body_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)  # 身体特征维度

        # 增加 Conv1D 融合模块
        self.fusion_conv1d = nn.Conv1d(in_channels=2 * code_dim, out_channels=code_dim, kernel_size=1)

        self.body_quantizer = QuantizeEMA_h2vq(nb_code_body, code_dim, args)

        # 手部特征投影层，用于映射手部特征到身体编码空间
        self.hand_projection = nn.Linear(code_dim, code_dim)
        
        self.decode_projection = nn.Linear(code_dim, code_dim)

        # 解码器，用于从融合后的手部和身体特征生成全身动作
        self.decoder = Decoder(self.whole_body, output_emb_width * 2, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def get_motion_from_gt(self, motion):
        pass

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        # (bs, Jx3, T) -> (bs, T, Jx3)
        return x.permute(0, 2, 1)

    def encode_hand(self, x, mask):
        x = self.preprocess(x)
        x_encoded = self.hand_encoder(x, motion_mask=mask)  # 手部动作无需额外 mask
        # x_encoded = x_encoded.contiguous().view(-1, x_encoded.shape[-1])
        x_quantized, loss, perplexity, hand_code_idx  = self.hand_quantizer(x_encoded)
        # x_encoded = self.postprocess(x_encoded)
        # # x_quantized, loss, perplexity  = self.hand_quantizer(x_encoded)

        # x_encoded = x_encoded.contiguous().view(-1, x_encoded.shape[-1])
        # hand_code_idx = self.hand_quantizer.quantize(x_encoded)
        # hand_code_idx = hand_code_idx.view(x.size(0), -1)  # 恢复 batch 维度
        return x_quantized, loss, perplexity, hand_code_idx

    def encode_body(self, x, hand_code_idx, mask):
        x = self.preprocess(x)
        x_encoded = self.body_encoder(x, motion_mask=mask)
        N, T, DIM = x_encoded.shape
        x_encoded = self.postprocess(x_encoded)
        # 融合手部特征
        hand_features = self.hand_quantizer.dequantize(hand_code_idx)
        hand_features = self.hand_projection(hand_features)  # 手部特征投影
        hand_features = hand_features.view(N, T, -1).permute(0, 2, 1).contiguous() # (N, DIM, T)
        # hand_features = hand_features.unsqueeze(1).expand(-1, x_encoded.size(1), -1)  # 扩展到时间步维度
        fused_features = torch.cat([x_encoded, hand_features], dim=-1)  # 特征拼接

        # 使用 Conv1D 融合
        fused_features = fused_features.permute(0, 2, 1)  # 调整形状以匹配 Conv1D 输入 (bs, channels, time)
        fused_features = self.fusion_conv1d(fused_features)  # 应用 1D 卷积
        fused_features = fused_features.permute(0, 2, 1)  # 恢复形状 (bs, time, channels)

        x_encoded = self.postprocess(fused_features)  # x_encoded.shape = (bs, time, channels)
        x_quantized, loss, perplexity, body_code_idx  = self.body_quantizer(x_encoded)

        # x_encoded = self.body_quantizer.preprocess(x_encoded)  # 预处理以匹配 quantize 输入要求
        # body_code_idx = self.body_quantizer.quantize(x_encoded)

        # body_code_idx = body_code_idx.view(x.size(0), -1)
        return x_quantized, loss, perplexity, body_code_idx

    def decode(self, hand_codes, body_codes, mask):
        N, _ = mask.shape
        # 将离散代码反量化
        hand_codes = hand_codes.view(N, -1).contiguous()
        body_codes = body_codes.view(N, -1).contiguous()
        hand_features = self.hand_quantizer.dequantize(hand_codes)
        body_features = self.body_quantizer.dequantize(body_codes)
        # 融合手部和身体特征
        fused_features = torch.cat([body_features, hand_features], dim=-1)
        # fused_features = fused_features.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        fused_features = fused_features.permute(0, 2, 1).contiguous()  # (N, DIM, T)

        # 解码生成全身动作
        x_decoder = self.decoder(fused_features, motion_mask=mask)
        
        return self.postprocess(x_decoder)

    def forward(self, whole_body_input, mask):
        # 编码手部动作
        motion_body = torch.cat((whole_body_input[:,:,:66], whole_body_input[:,:,66+90:]), dim=2).clone()
        motion_hand = whole_body_input[:,:,66:66+90].clone()

        hand_quantized, hand_loss, hand_perplexity, hand_code_idx = self.encode_hand(motion_hand, mask)
        
        body_quantized, body_loss, body_perplexity, body_code_idx = self.encode_body(motion_body, hand_code_idx, mask)

        # 解码生成全身动作
        reconstructed_motion = self.decode(hand_code_idx, body_code_idx, mask)
        reconstructed_hand = reconstructed_motion[:,:,-self.hand_dim:].clone()
        reconstructed_body = reconstructed_motion[:,:,:self.body_dim].clone()
        
        motion_decode = torch.zeros_like(reconstructed_motion)
        motion_decode[:,:,66:66+90] = reconstructed_hand
        motion_decode[:,:,:66] = reconstructed_body[:,:,:66]
        motion_decode[:,:,66+90:] = reconstructed_body[:,:,66:]

        total_loss = hand_loss + body_loss
        avg_perplexity = (hand_perplexity + body_perplexity) / 2

        return motion_decode, total_loss, avg_perplexity


# @SUBMODULES.register_module()
class H2VQ_bodybase_hand(nn.Module):
    def __init__(self,
                 args,
                #  nb_code_hand=512,
                 nb_code_body=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                #  face_dim = 150,
                #  hand_dim = 90,
                 body_dim = 82,
                 norm=None):
        super().__init__()

        self.code_dim = code_dim
        self.nb_code_body = nb_code_body
        self.body_dim = body_dim
        # self.whole_body = self.hand_dim + self.body_dim

        # 初始化身体编码器和量化器
        self.body_encoder = Encoder(self.body_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)  # 身体特征维度

        # 增加 Conv1D 融合模块
        # self.fusion_conv1d = nn.Conv1d(in_channels=2 * code_dim, out_channels=code_dim, kernel_size=1)

        self.body_quantizer = QuantizeEMA_h2vq(nb_code_body, code_dim, args)

        # 手部特征投影层，用于映射手部特征到身体编码空间
        # self.hand_projection = nn.Linear(code_dim, code_dim)

        # 解码器，用于从融合后的手部和身体特征生成全身动作
        self.decoder = Decoder(self.body_dim, output_emb_width * 2, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)

    def get_motion_from_gt(self, motion):
        pass

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        return x.permute(0, 2, 1).float()

    def postprocess(self, x):
        # (bs, Jx3, T) -> (bs, T, Jx3)
        return x.permute(0, 2, 1)

    def encode(self, x, mask):
        x = self.preprocess(x)
        x_encoded = self.hand_encoder(x, motion_mask=mask)  # 手部动作无需额外 mask
        # x_encoded = x_encoded.contiguous().view(-1, x_encoded.shape[-1])
        x_quantized, loss, perplexity, hand_code_idx  = self.hand_quantizer(x_encoded)
        # x_encoded = self.postprocess(x_encoded)
        # # x_quantized, loss, perplexity  = self.hand_quantizer(x_encoded)

        # x_encoded = x_encoded.contiguous().view(-1, x_encoded.shape[-1])
        # hand_code_idx = self.hand_quantizer.quantize(x_encoded)
        # hand_code_idx = hand_code_idx.view(x.size(0), -1)  # 恢复 batch 维度
        return x_quantized, loss, perplexity, hand_code_idx

    # def encode_body(self, x, hand_code_idx, mask):
    #     x = self.preprocess(x)
    #     x_encoded = self.body_encoder(x, motion_mask=mask)
    #     N, T, DIM = x_encoded.shape
    #     x_encoded = self.postprocess(x_encoded)
    #     # 融合手部特征
    #     hand_features = self.hand_quantizer.dequantize(hand_code_idx)
    #     hand_features = self.hand_projection(hand_features)  # 手部特征投影
    #     hand_features = hand_features.view(N, T, -1).permute(0, 2, 1).contiguous() # (N, DIM, T)
    #     # hand_features = hand_features.unsqueeze(1).expand(-1, x_encoded.size(1), -1)  # 扩展到时间步维度
    #     fused_features = torch.cat([x_encoded, hand_features], dim=-1)  # 特征拼接

    #     # 使用 Conv1D 融合
    #     fused_features = fused_features.permute(0, 2, 1)  # 调整形状以匹配 Conv1D 输入 (bs, channels, time)
    #     fused_features = self.fusion_conv1d(fused_features)  # 应用 1D 卷积
    #     fused_features = fused_features.permute(0, 2, 1)  # 恢复形状 (bs, time, channels)

    #     x_encoded = self.postprocess(fused_features)  # x_encoded.shape = (bs, time, channels)
    #     x_quantized, loss, perplexity, body_code_idx  = self.body_quantizer(x_encoded)

    #     return x_quantized, loss, perplexity, body_code_idx

    def decode(self, hand_codes, body_codes, mask):
        N, _ = mask.shape
        # 将离散代码反量化
        hand_codes = hand_codes.view(N, -1).contiguous()
        body_codes = body_codes.view(N, -1).contiguous()
        hand_features = self.hand_quantizer.dequantize(hand_codes)
        body_features = self.body_quantizer.dequantize(body_codes)
        # 融合手部和身体特征
        fused_features = torch.cat([hand_features, body_features], dim=-1)
        # fused_features = fused_features.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        fused_features = fused_features.permute(0, 2, 1).contiguous()  # (N, DIM, T)

        # 解码生成全身动作
        x_decoder = self.decoder(fused_features, motion_mask=mask)

        return self.postprocess(x_decoder)

    def forward(self, whole_body_input, mask):
        # 编码手部动作
        motion_body = torch.cat((whole_body_input[:,:,:66], whole_body_input[:,:,66+90:]), dim=2).clone()
        motion_hand = whole_body_input[:,:,66:66+90].clone()

        hand_quantized, hand_loss, hand_perplexity, hand_code_idx = self.encode_hand(motion_hand, mask)
        
        body_quantized, body_loss, body_perplexity, body_code_idx = self.encode_body(motion_body, hand_code_idx, mask)

        # 解码生成全身动作
        reconstructed_motion = self.decode(hand_code_idx, body_code_idx, mask)

        total_loss = hand_loss + body_loss
        avg_perplexity = (hand_perplexity + body_perplexity) / 2

        return reconstructed_motion, total_loss, avg_perplexity
