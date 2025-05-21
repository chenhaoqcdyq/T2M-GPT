import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding
from einops import rearrange, repeat
import math
from random import random
from tqdm.auto import tqdm
from typing import Callable, Optional, List, Dict
from copy import deepcopy
from functools import partial
import clip
from models.encdec import CausalTransformerEncoder
from einops import rearrange

class Residual_encoder(nn.Module):
    def __init__(self, t2m_encoder, residual_encoder, embed_dim, num_vq=512, semantic_flag=False, num_quantizers = 6, clip_dim=512):
        super().__init__()
        self.t2m_encoder = t2m_encoder
        self.residual_encoder = residual_encoder
        self.num_quantizers = num_quantizers
        self.tok_emb = nn.ModuleList([nn.Embedding(num_vq + 2, embed_dim) for _ in range(num_quantizers)])
        self.semantic_flag = semantic_flag
        self.clip_emb = nn.ModuleList([nn.Linear(clip_dim, embed_dim) for _ in range(num_quantizers)])
        # self.token_emb = nn.Linear(num_vq + 1, embed_dim)
        self.embed_dim = embed_dim
        if self.semantic_flag:
            self.sem_tok_emb = nn.ModuleList([nn.Embedding(num_vq + 2, embed_dim) for _ in range(num_quantizers)])
            self.sem_len = t2m_encoder.semantic_len
        
        
    def forward(self, a_indices, feat_clip_text, sem_tokens_len=None):
        # a_indices (B, P, L)
        if self.semantic_flag:
            sem_feature = [self.sem_tok_emb[i](a_indices[:, i:i+1, :self.sem_len]) for i in range(self.num_quantizers)]
            sem_feature = torch.cat(sem_feature, dim=1)
            recon_feature = [self.tok_emb[i](a_indices[:, i:i+1, self.sem_len:]) for i in range(self.num_quantizers)]
            recon_feature = torch.cat(recon_feature, dim=1)
            feature_a_indices = torch.cat([sem_feature, recon_feature], dim=2)
        else:
            feature_a_indices = [self.tok_emb[i](a_indices[:, i:i+1, :]) for i in range(self.num_quantizers)]
            # (B, L, C) * P -> (B, P, L, C)
            feature_a_indices = torch.cat(feature_a_indices,dim=1)
        # feature_a_indices (B, P, L, C)
        first_quantizer_indices = feature_a_indices[:, 0, :, :]
        if sem_tokens_len is not None:
            first_quantizer_cls_pred = self.t2m_encoder(first_quantizer_indices, feat_clip_text, semantic_valid_lengths=sem_tokens_len)
        else:
            first_quantizer_cls_pred = self.t2m_encoder(first_quantizer_indices, feat_clip_text)
        first_quantizer_cls_pred = first_quantizer_cls_pred.contiguous() 
        # first_quantizer_cls_pred (B, P, C)
        bs, L, D = first_quantizer_cls_pred.shape
        first_quantizer_cls_pred = rearrange(first_quantizer_cls_pred, 'b l c -> (b l) c').unsqueeze(1)
        # feature_a_indices_woclip = rearrange(feature_a_indices[:, :P-1, :, :], 'b p l c -> (b l) p c')
        feature_a_indices_wclip = torch.stack([torch.cat([self.clip_emb[i](feat_clip_text).unsqueeze(1), feature_a_indices[:, i, :, :]], dim=1) for i in range(self.num_quantizers)], dim=1)
        residual_input = torch.cat([first_quantizer_cls_pred, rearrange(feature_a_indices_wclip, 'b p l c -> (b l) p c')], dim=1)
        residual_cls_pred = self.residual_encoder(torch.cumsum(residual_input[:, :self.num_quantizers, :], dim=1))
        # residual_cls_pred (B*L, P-1, C)
        # cls_pred = torch.cumsum(residual_cls_pred, dim=1)
        # cls_pred (B*L, P, C)
        cls_pred = rearrange(residual_cls_pred, '(b l) p c -> b l p c', b=bs, l=L)

        return cls_pred

    def sample(self, clip_feature, if_categorial=False, cfg_scale=7.5, use_cfg=False):
        """
        自回归生成残差索引，按时间步和层次顺序生成
        
        参数:
            clip_feature: 文本特征 [batch_size, clip_dim]
            if_categorial: 是否使用分类采样
            cfg_scale: CFG引导强度，仅在use_cfg=True时使用
            use_cfg: 是否使用条件引导生成
            
        返回:
            生成的token序列 [batch_size, num_quantizers, seq_len]
        """
        device = clip_feature.device
        batch_size = clip_feature.shape[0]
        
        # 初始化所有量化器的token序列
        max_seq_len = self.t2m_encoder.block_size
        all_tokens = torch.zeros(batch_size, self.num_quantizers, max_seq_len, dtype=torch.long, device=device)
        
        # 记录序列是否结束的标志
        all_sequences_finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        # 当前序列长度
        current_seq_len = 0
        
        # 记录语义部分的有效长度（如果启用语义标志）
        semantic_valid_lengths = None
        if self.semantic_flag:
            semantic_valid_lengths = torch.full((batch_size,), self.sem_len, dtype=torch.long, device=device)
            # 获取语义和重建token的分界点
            _assumed_codebook_part_size = (self.t2m_encoder.num_vq - 1) // 2
            semantic_token_end_idx = _assumed_codebook_part_size - 1
            separator_token_idx = _assumed_codebook_part_size
            reconstruction_token_start_idx = _assumed_codebook_part_size + 1
        
        # 逐时间步生成
        while current_seq_len < max_seq_len and not all_sequences_finished.all():
            # 准备当前时间步的输入
            if current_seq_len == 0:
                # 第一个时间步，没有历史token
                xs = torch.empty((batch_size, 0, self.embed_dim), dtype=torch.long, device=device)
            else:
                # 使用之前生成的第一层token作为输入
                if self.semantic_flag and current_seq_len < self.sem_len:
                    xs = self.sem_tok_emb[0](all_tokens[:, 0, :current_seq_len])
                elif self.semantic_flag and current_seq_len >= self.sem_len:
                    xs = torch.cat([self.sem_tok_emb[0](all_tokens[:, 0, :self.sem_len]), self.tok_emb[0](all_tokens[:, 0, self.sem_len:current_seq_len])], dim=1)
                else:
                    xs = self.tok_emb[0](all_tokens[:, 0, :current_seq_len])
            
            # 1. 使用t2m_encoder生成当前时间步的特征
            first_quantizer_features = self.t2m_encoder(xs, clip_feature, semantic_valid_lengths)
            first_quantizer_features = first_quantizer_features.contiguous()
            
            # 只取最后一个时间步的特征用于预测当前时间步的token
            bs, L, D = first_quantizer_features.shape
            last_step_feature = first_quantizer_features[:, -1:, :]  # [B, 1, D]
            
            # 2. 重塑为residual_encoder所需的输入格式，参考forward函数
            first_quantizer_features_reshaped = rearrange(last_step_feature, 'b l c -> (b l) c').unsqueeze(1)
            
            # 3. 逐层生成当前时间步的token
            for layer_idx in range(self.num_quantizers):
                # 准备当前层的输入
                if layer_idx == 0:
                    # 第一层
                    # 添加CLIP特征
                    clip_emb = self.clip_emb[layer_idx](clip_feature).unsqueeze(1)
                    
                    # 构建feature_a_indices_wclip
                    feature_a_indices_wclip = torch.stack([clip_emb], dim=1)  # [B, 1, 1, D]
                    
                    # 合并输入
                    residual_input = torch.cat([
                        first_quantizer_features_reshaped,  # [(B*1), 1, D]
                        rearrange(feature_a_indices_wclip, 'b p l c -> (b l) p c')  # [(B*1), 1, D]
                    ], dim=1)
                    
                    # 累积求和
                    residual_input = torch.cumsum(residual_input[:, :1, :], dim=1)
                else:
                    # 获取当前时间步已生成的层的token
                    current_step_tokens = all_tokens[:, :layer_idx, current_seq_len]  # [B, layer_idx]
                    
                    # 嵌入这些token
                    current_step_embeds = []
                    for l in range(layer_idx):
                        # 根据当前是否在语义部分选择不同的嵌入
                        if self.semantic_flag and current_seq_len < self.sem_len:
                            # 语义部分使用语义嵌入
                            token_emb = self.sem_tok_emb[l](current_step_tokens[:, l]).unsqueeze(1)  # [B, 1, D]
                        else:
                            # 重建部分使用普通嵌入
                            token_emb = self.tok_emb[l](current_step_tokens[:, l]).unsqueeze(1)  # [B, 1, D]
                        current_step_embeds.append(token_emb)
                    
                    # 添加CLIP特征
                    clip_emb = self.clip_emb[layer_idx](clip_feature).unsqueeze(1)  # [B, 1, D]
                    
                    # 合并所有特征
                    feature_list = [clip_emb] + current_step_embeds  # [[B, 1, D], ...]
                    feature_a_indices_wclip = torch.stack(feature_list, dim=1)  # [B, layer_idx+1, 1, D]
                    
                    # 合并输入
                    residual_input = torch.cat([
                        first_quantizer_features_reshaped,  # [(B*1), 1, D]
                        rearrange(feature_a_indices_wclip, 'b p l c -> (b l) p c')  # [(B*1), layer_idx+1, D]
                    ], dim=1)
                    
                    # 累积求和
                    residual_input = torch.cumsum(residual_input[:, :layer_idx+1, :], dim=1)
                
                # 4. 使用residual_encoder预测
                residual_cls_pred = self.residual_encoder(residual_input)  # [(B*1), layer_idx, vocab_size]
                
                # 5. 获取当前层的logits
                logits = residual_cls_pred[:, -1, :]  # [(B*1), vocab_size]
                
                # 6. 如果启用语义标志，根据当前位置修改logits
                if self.semantic_flag and layer_idx == 0:  # 只对第一层进行处理
                    if current_seq_len < self.sem_len:
                        # 在语义部分，屏蔽重建token和EOS
                        logits[:, reconstruction_token_start_idx:] = float('-inf')
                        
                        # 检查是否到达语义部分末尾
                        if current_seq_len == self.sem_len - 1:
                            # 在语义部分最后一个位置，强制使用分隔符
                            logits[:, :] = float('-inf')
                            logits[:, separator_token_idx] = 0.0
                    else:
                        # 在重建部分，屏蔽语义token和分隔符
                        logits[:, :reconstruction_token_start_idx+1] = float('-inf')
                
                # 7. 对logits进行采样
                if if_categorial:
                    probs = F.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    token = dist.sample()  # [(B*1)]
                else:
                    _, token = torch.topk(logits, k=1, dim=-1)
                    token = token.squeeze(-1)  # [(B*1)]
                
                # 8. 保存当前层的token
                all_tokens[:, layer_idx, current_seq_len] = token
                
                # 9. 如果是第一层，检查是否有序列结束或更新语义有效长度
                if layer_idx == 0:
                    for i in range(batch_size):
                        # 检查是否结束序列
                        if token[i] == self.t2m_encoder.num_vq:  # EOS token
                            all_sequences_finished[i] = True
                        
                        # 如果在语义部分且启用了语义标志，更新语义有效长度
                        if self.semantic_flag and current_seq_len < self.sem_len:
                            # 如果生成了EOS或分隔符，更新语义有效长度
                            if token[i] == self.t2m_encoder.num_vq or token[i] == separator_token_idx:
                                semantic_valid_lengths[i] = current_seq_len + 1
            
            # 更新序列长度
            current_seq_len += 1
        
        # 返回生成的token序列，截取到实际生成的长度
        return all_tokens[:, :, :current_seq_len]

    def sample_batch_with_cfg(self, clip_feature, if_categorial=False, cfg_scale=7.5):
        """
        批量条件引导采样函数，支持一批次中一半是有文本特征，一半是空字符串特征的情况
        
        参数:
            clip_feature: 形状为 [batch_size, clip_dim] 的文本特征
                          其中前半部分是正常文本特征，后半部分是空字符串特征
            if_categorial: 是否使用分类采样
            cfg_scale: CFG引导强度，正值越大引导越强，推荐范围 5.0-10.0
            
        返回:
            生成的token序列 [batch_size/2, num_quantizers, seq_len]
        """
        # 假设输入的clip_feature已经是[cond1, cond2, ..., uncond1, uncond2, ...]的形式
        full_batch_size = clip_feature.shape[0]
        assert full_batch_size % 2 == 0, "批次大小必须是偶数，一半条件一半无条件"
        
        half_batch_size = full_batch_size // 2
        device = clip_feature.device
        
        # 首先使用t2m_encoder生成第一层量化器的token序列
        first_layer_tokens = self.t2m_encoder.sample_cfg_batch_with_empty(clip_feature, if_categorial, cfg_scale)
        
        # 获取序列长度
        seq_len = first_layer_tokens.shape[1]
        
        # 初始化所有量化器的token序列
        all_tokens = torch.zeros(half_batch_size, self.num_quantizers, seq_len, dtype=torch.long, device=device)
        # 设置第一层量化器的token序列
        all_tokens[:, 0, :] = first_layer_tokens
        
        # 分离条件和无条件特征
        cond_features = clip_feature[:half_batch_size]
        uncond_features = clip_feature[half_batch_size:]
        
        # 记录语义部分的有效长度（如果启用语义标志）
        semantic_valid_lengths = None
        if self.semantic_flag:
            semantic_valid_lengths = torch.full((half_batch_size,), self.sem_len, dtype=torch.long, device=device)
            # 检查是否有提前结束的语义部分
            for b in range(half_batch_size):
                for t in range(min(self.sem_len, seq_len)):
                    if first_layer_tokens[b, t] == self.t2m_encoder.num_vq:  # 检测到结束标记
                        semantic_valid_lengths[b] = t
                        break
        
        # 逐时间步生成剩余量化器的token
        for t in range(seq_len):
            # 为每个时间步t逐层生成token（从第2层开始，因为第1层已经由t2m_encoder生成）
            for layer_idx in range(1, self.num_quantizers):
                # 准备当前已生成的token序列作为条件
                # 提取到当前时间步t的所有已生成token
                current_tokens = all_tokens[:, :layer_idx, :t+1]  # [B/2, layer_idx, t+1]
                
                # 为当前层创建嵌入特征
                feature_list = []
                for i in range(layer_idx):
                    if self.semantic_flag and i == 0:  # 第一层使用语义嵌入
                        # 处理语义部分，需要考虑padding
                        if t < self.sem_len:  # 在语义部分范围内
                            layer_emb = self.sem_tok_emb[i](current_tokens[:, i, :])
                        else:  # 超出语义部分，使用普通嵌入
                            layer_emb = self.tok_emb[i](current_tokens[:, i, :])
                    else:
                        layer_emb = self.tok_emb[i](current_tokens[:, i, :])
                    feature_list.append(layer_emb)
                
                # 将所有层的嵌入特征合并
                combined_features = torch.stack(feature_list, dim=1)  # [B/2, layer_idx, t+1, embed_dim]
                
                # 对条件和无条件特征分别处理
                # 条件部分
                cond_clip_features = [self.clip_emb[i](cond_features).unsqueeze(1) for i in range(layer_idx)]
                cond_clip_features = torch.stack(cond_clip_features, dim=1)  # [B/2, layer_idx, 1, embed_dim]
                
                # 无条件部分
                uncond_clip_features = [self.clip_emb[i](uncond_features).unsqueeze(1) for i in range(layer_idx)]
                uncond_clip_features = torch.stack(uncond_clip_features, dim=1)  # [B/2, layer_idx, 1, embed_dim]
                
                # 合并CLIP特征和token嵌入
                cond_input = torch.cat([cond_clip_features, combined_features], dim=2)
                uncond_input = torch.cat([uncond_clip_features, combined_features], dim=2)
                
                # 创建padding mask，如果在语义部分且有提前结束的token
                key_padding_mask = None
                if self.semantic_flag and t < self.sem_len and semantic_valid_lengths is not None:
                    key_padding_mask = torch.zeros(half_batch_size, t+2, dtype=torch.bool, device=device)  # +2是因为包含了clip特征
                    for b in range(half_batch_size):
                        if semantic_valid_lengths[b] < t:  # 如果当前时间步超过了有效语义长度
                            # 将超出有效长度的部分标记为padding
                            key_padding_mask[b, semantic_valid_lengths[b]+1:t+1] = True
                
                # 重塑为residual_encoder所需的输入格式
                bs, layers, seq_plus_one, dim = cond_input.shape
                cond_residual_input = rearrange(cond_input, 'b p l c -> (b l) p c')
                uncond_residual_input = rearrange(uncond_input, 'b p l c -> (b l) p c')
                
                # 累积求和以获取层间依赖关系
                cond_cumsum_input = torch.cumsum(cond_residual_input, dim=1)
                uncond_cumsum_input = torch.cumsum(uncond_residual_input, dim=1)
                
                # 使用residual_encoder预测当前时间步t的第layer_idx层token
                cond_residual_logits = self.residual_encoder(cond_cumsum_input)  # [(B/2*(t+2)), 1, vocab_size]
                uncond_residual_logits = self.residual_encoder(uncond_cumsum_input)  # [(B/2*(t+2)), 1, vocab_size]
                
                # 重塑logits以便于采样，只取最后一个位置（当前时间步）的logits
                cond_logits = rearrange(cond_residual_logits, '(b l) p v -> b l p v', b=bs)[:, -1, 0, :]  # [B/2, vocab_size]
                uncond_logits = rearrange(uncond_residual_logits, '(b l) p v -> b l p v', b=bs)[:, -1, 0, :]  # [B/2, vocab_size]
                
                # 应用CFG公式: logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
                cfg_logits = uncond_logits + cfg_scale * (cond_logits - uncond_logits)
                
                # 对logits进行批量采样
                if if_categorial:
                    probs = F.softmax(cfg_logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    sampled_token = dist.sample()  # [B/2]
                else:
                    _, sampled_token = torch.topk(cfg_logits, k=1, dim=-1)
                    sampled_token = sampled_token.squeeze(-1)  # [B/2]
                
                # 将当前层的token添加到结果中
                all_tokens[:, layer_idx, t] = sampled_token
        
        return all_tokens

class Text2Motion_Transformer(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_flag=False,
                semantic_interleaved_flag=False,
                dual_head_flag=False,
                semantic_len=50):
        super().__init__()
        
        if dual_head_flag:
            self.trans_base = CrossCondTransDualBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, semantic_len)
            # self.trans_head = CrossCondTransDualHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate, semantic_len)
        else:
            # self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
            self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq
        self.semantic_flag = semantic_flag
        self.semantic_interleaved_flag = semantic_interleaved_flag
        self.dual_head_flag = dual_head_flag
        self.semantic_len = semantic_len

        _assumed_codebook_part_size = (self.num_vq - 1) // 2 # e.g. (1025-1)//2 = 512

        self.SEMANTIC_TOKEN_END_IDX = _assumed_codebook_part_size - 1
        self.SEPARATOR_TOKEN_IDX = _assumed_codebook_part_size
        self.RECONSTRUCTION_TOKEN_START_IDX = _assumed_codebook_part_size + 1
        # RECONSTRUCTION_TOKEN_END_IDX is self.num_vq - 1 (the max token ID before stop)
        # STOP_TOKEN_ID is self.num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature, semantic_valid_lengths=None):
        # idxs: (B, T_sequence), contains semantic (potentially padded) and reconstruction tokens
        # clip_feature: (B, C_clip)
        # semantic_valid_lengths: (B,), actual length of semantic tokens in idxs for each batch item.
        # self.semantic_len: max length of the semantic part within idxs.
        if idxs is not None:
            B, T_sequence, dim = idxs.shape
        else:
            B, T_sequence, dim = clip_feature.shape
        device = idxs.device

        # 1. Create key_padding_mask for the idxs part based on semantic_valid_lengths
        # This mask is True for padded semantic tokens, False otherwise (valid semantic, reconstruction).
        key_padding_mask_for_idxs = torch.zeros_like(idxs[:, :, 0], dtype=torch.bool) # Default to False (not masked)

        if semantic_valid_lengths is not None:
            for i in range(B):
                valid_sem_len = semantic_valid_lengths[i].item()
                # Determine the end of the semantic segment within idxs.
                # This is typically self.semantic_len if idxs is long enough to contain it.
                # Or it could be T_sequence if idxs is shorter than self.semantic_len.
                # The padding occurs *within* the first self.semantic_len tokens of idxs.
                actual_semantic_segment_end = self.semantic_len
                
                if valid_sem_len < actual_semantic_segment_end:
                    # Mask padded semantic tokens: from valid_sem_len to actual_semantic_segment_end
                    key_padding_mask_for_idxs[i, valid_sem_len:actual_semantic_segment_end] = True
        
        # 2. Prepare the final attention mask for the sequence seen by attention layers in trans_head.
        # This sequence is [condition_embedding, token_embeddings_from_idxs].
        final_attention_mask_for_trans_head_input = torch.cat([
            torch.zeros(B, 1, dtype=torch.bool, device=device), # False for condition embedding
            key_padding_mask_for_idxs
        ], dim=1)

        # Pass the appropriate masks to trans_base and trans_head
        feat = self.trans_base(idxs, clip_feature, key_padding_mask=final_attention_mask_for_trans_head_input)
        # logits = self.trans_head(feat, key_padding_mask=final_attention_mask_for_trans_head_input)
        
        return feat

    def sample_dual_head(self, clip_feature, if_categorial=False):
        B = clip_feature.shape[0]
        device = clip_feature.device
        xs = None 
        
        # semantic_content_has_ended_flags[i] is True if self.num_vq was sampled (signaling end of semantic data)
        # for batch item i, leading to subsequent actual padding tokens (self.num_vq + 1).
        # semantic_content_has_ended_flags = torch.zeros(B, dtype=torch.bool, device=device)
        
        # Stores the actual length of semantic tokens before any (self.num_vq + 1) padding started.
        # Initialized to self.semantic_len, assuming full semantic part unless ended early.
        actual_semantic_tokens_count = torch.full((B,), self.semantic_len, dtype=torch.long, device=device)

        for _ in range(self.block_size): 
            current_len = xs.shape[1] if xs is not None else 0
            
            if current_len >= self.block_size:
                break

            current_input_for_transformer = xs 
            if current_input_for_transformer is None:
                 current_input_for_transformer = torch.empty(B, 0, dtype=torch.long, device=device)
            
            # Determine semantic_valid_lengths for the forward pass.
            # Only pass it if current_len is sufficient for the forward method's masking logic to apply.
            fwd_semantic_valid_lengths = None
            if current_len >= self.semantic_len:
                # Pass the actual semantic token counts. forward will use this to mask
                # padding tokens (self.num_vq + 1) that might be within the first self.semantic_len block of xs.
                fwd_semantic_valid_lengths = actual_semantic_tokens_count.clone()
            
            logits = self.forward(current_input_for_transformer, clip_feature, semantic_valid_lengths=fwd_semantic_valid_lengths)
            logits = logits[:, -1, :] 
            # probs = F.softmax(logits, dim=-1) # (B, V)

            next_step_tokens_for_batch = torch.zeros(B, dtype=torch.long, device=device)

            if current_len < self.semantic_len: # Processing semantic part
                candidate_tokens_this_step = None # Shape (B,)
                # if if_categorial:
                #     dist = Categorical(probs)
                #     candidate_tokens_this_step = dist.sample() 
                # else:
                #     _, topk_tokens = torch.topk(probs, k=1, dim=-1) 
                #     candidate_tokens_this_step = topk_tokens.squeeze(-1)

                for i in range(B):
                    if semantic_content_has_ended_flags[i]:
                        # Semantic content for this item ended, force padding (self.num_vq + 1)
                        next_step_tokens_for_batch[i] = self.num_vq + 1
                    else:
                        token_for_item_i = candidate_tokens_this_step[i]
                        if token_for_item_i == self.num_vq: # Sampled self.num_vq, signaling end of true semantic content
                            semantic_content_has_ended_flags[i] = True 
                            actual_semantic_tokens_count[i] = current_len # True semantic data length is current_len
                            next_step_tokens_for_batch[i] = self.num_vq # Start padding from now on
                        else: # Regular semantic token
                            next_step_tokens_for_batch[i] = token_for_item_i
                
                idx_sampled_token_tensor = next_step_tokens_for_batch.unsqueeze(-1) # (B,1)

            else: # Processing reconstruction part (current_len >= self.semantic_len)
                # Ensure flags and counts are accurate for items that filled self.semantic_len without an explicit stop signal
                for i in range(B):
                    if not semantic_content_has_ended_flags[i]:
                        semantic_content_has_ended_flags[i] = True
                        # actual_semantic_tokens_count[i] remains self.semantic_len (its initial value)
                        # if it reached here without semantic_content_has_ended_flags[i] being True.

                # Original sampling logic for reconstruction part:
                idx_sampled_token_tensor = None 
                if if_categorial:
                    dist = Categorical(probs)
                    idx_val_batch = dist.sample() # (B,)
                    # Simplified batch stop: if first item stops, all stop.
                    # A robust batch solution would mask completed sequences.
                    if idx_val_batch[0] == self.num_vq: 
                        break 
                    idx_sampled_token_tensor = idx_val_batch.unsqueeze(-1)
                else:
                    _, idx_val_topk_batch = torch.topk(probs, k=1, dim=-1) # (B,1)
                    if idx_val_topk_batch[0,0] == self.num_vq: 
                        break 
                    idx_sampled_token_tensor = idx_val_topk_batch
            
            if idx_sampled_token_tensor is None: # Should only happen if loop broke in recon phase
                break

            if xs is None:
                xs = idx_sampled_token_tensor
            else:
                xs = torch.cat((xs, idx_sampled_token_tensor), dim=1)
                
        if xs is None: 
            return torch.empty((B, 0), dtype=torch.long, device=clip_feature.device)
        return xs
    
    def sample_original_backup(self, clip_feature, if_categorial=False):
        """
        Original sampling logic, used when semantic_flag=False.
        Samples from the full vocabulary space. self.num_vq is the stop token.
        """
        xs = None
        for k_loop_idx in range(self.block_size):
            current_input_for_transformer = xs if xs is not None else [] 
            
            logits = self.forward(current_input_for_transformer, clip_feature)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)

            idx_sampled_token = None
            if if_categorial:
                dist = Categorical(probs)
                idx_val = dist.sample() 
                if idx_val == self.num_vq: 
                    break 
                idx_sampled_token = idx_val.unsqueeze(-1) 
            else:
                _, idx_val_topk = torch.topk(probs, k=1, dim=-1) 
                if idx_val_topk[0,0] == self.num_vq: 
                    break 
                idx_sampled_token = idx_val_topk
            
            if xs is None:
                xs = idx_sampled_token
            else:
                xs = torch.cat((xs, idx_sampled_token), dim=1)
            
        if xs is None: 
            return torch.empty((clip_feature.shape[0], 0), dtype=torch.long, device=clip_feature.device)
        return xs


    def sample(self, clip_feature, if_categorial=False):
        if self.dual_head_flag:
            return self.sample_dual_head(clip_feature, if_categorial)
        else:
            return self.sample_original_backup(clip_feature, if_categorial)

    
class Text2Motion_RVQ_Transformer(nn.Module):
    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=7, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq

        # RECONSTRUCTION_TOKEN_END_IDX is self.num_vq - 1 (the max token ID before stop)
        # STOP_TOKEN_ID is self.num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature=None):
        # B, T_sequence = idxs.shape
        # device = idxs.device
        # Pass the appropriate masks to trans_base and trans_head
        feat = self.trans_base(idxs, clip_feature)
        logits = self.trans_head(feat)
        
        return logits

    def sample_original_backup(self, clip_feature, if_categorial=False):
        xs = clip_feature
        for k_loop_idx in range(self.block_size):
            current_input_for_transformer = xs if xs is not None else [] 
            
            logits = self.forward(xs)
            logits = logits[:, -1, :] 
            probs = F.softmax(logits, dim=-1)

            idx_sampled_token = None
            if if_categorial:
                dist = Categorical(probs)
                idx_val = dist.sample() 
                if idx_val == self.num_vq: 
                    break 
                idx_sampled_token = idx_val.unsqueeze(-1) 
            else:
                _, idx_val_topk = torch.topk(probs, k=1, dim=-1) 
                if idx_val_topk[0,0] == self.num_vq: 
                    break 
                idx_sampled_token = idx_val_topk
            
            if xs is None:
                xs = idx_sampled_token
            else:
                xs = torch.cat((xs, idx_sampled_token), dim=1)
            
        if xs is None: 
            return torch.empty((clip_feature.shape[0], 0), dtype=torch.long, device=clip_feature.device)
        return xs

    def sample(self, clip_feature, if_categorial=False):
        return self.sample_original_backup(clip_feature, if_categorial)

    def sample_batch(self, clip_feature, if_categorial=False):
        B = clip_feature.shape[0]
        device = clip_feature.device
        xs = torch.empty((B, 0), dtype=torch.long, device=device)
        all_sequences_finished = torch.zeros(B, dtype=torch.bool, device=device)
        
        for loop_idx in range(self.block_size):
            if all_sequences_finished.all():
                break
            
            current_len = xs.shape[1] # Length of sequences before adding current step's tokens
            fwd_semantic_valid_lengths = None

            # current_input_for_transformer is xs. If xs is (B,0), forward handles it.
            logits = self.forward(xs, clip_feature, semantic_valid_lengths=fwd_semantic_valid_lengths)
            # Get logits for the next token prediction
            probs = F.softmax(logits[:, -1, :], dim=-1) 
            
            candidate_tokens = None
            if if_categorial:
                dist = Categorical(probs) # Probs: (B, V)
                candidate_tokens = dist.sample() # Cand: (B,)
            else:
                _, candidate_tokens_topk = torch.topk(probs, k=1, dim=-1) # Cand_topk: (B, 1)
                candidate_tokens = candidate_tokens_topk.squeeze(-1) # Cand: (B,)
                
            step_tokens = torch.zeros(B, dtype=torch.long, device=device)

            for i in range(B):
                if all_sequences_finished[i]:
                    step_tokens[i] = self.num_vq + 1 # PAD token
                    continue

                token_i = candidate_tokens[i]
                
                if token_i == self.num_vq: # EOS
                    step_tokens[i] = self.num_vq
                    all_sequences_finished[i] = True
                else:
                    step_tokens[i] = token_i
            
            xs = torch.cat((xs, step_tokens.unsqueeze(-1)), dim=1)
            
        return xs

class CausalCrossConditionalSelfAttention(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1):
        super().__init__()
        assert embed_dim % 8 == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

        self.attn_drop = nn.Dropout(drop_out_rate)
        self.resid_drop = nn.Dropout(drop_out_rate)

        self.proj = nn.Linear(embed_dim, embed_dim)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))
        self.n_head = n_head

    def forward(self, x, key_padding_mask=None):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # Apply causal mask to ensure attention is only to the left (and current position)
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
        
        # Apply key_padding_mask to prevent attention to padded key positions
        if key_padding_mask is not None:
            # key_padding_mask has shape (B, T) or (B, T_key) where T_key == T in self-attention
            # It should be True for positions that are padded and should be masked.
            # We need to expand it to (B, 1, 1, T) to be broadcastable with att (B, nh, T, T)
            expanded_key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)
            att = att.masked_fill(expanded_key_padding_mask, float('-inf'))
            
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block(nn.Module):

    def __init__(self, embed_dim=512, block_size=16, n_head=8, drop_out_rate=0.1, fc_rate=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = CausalCrossConditionalSelfAttention(embed_dim, block_size, n_head, drop_out_rate)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, fc_rate * embed_dim),
            nn.GELU(),
            nn.Linear(fc_rate * embed_dim, embed_dim),
            nn.Dropout(drop_out_rate),
        )

    def forward(self, x, key_padding_mask=None):
        x = x + self.attn(self.ln1(x), key_padding_mask=key_padding_mask)
        x = x + self.mlp(self.ln2(x))
        return x

class CrossCondTransBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()
        # self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        # transformer block
        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature=None):
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t, dim = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = idx
            if clip_feature is not None:
                token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

        return x

class CrossCondTransDualBase(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                clip_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_len = 50):
        super().__init__()
        # self.tok_emb = nn.ModuleList([nn.Embedding(num_vq + 2, embed_dim) for _ in range(2)])
        self.cond_emb = nn.Linear(clip_dim, embed_dim)
        self.pos_embedding = nn.Embedding(block_size, embed_dim)
        self.drop = nn.Dropout(drop_out_rate)
        self.semantic_len = semantic_len
        # transformer block
        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.pos_embed = pos_encoding.PositionEmbedding(block_size, embed_dim, 0.0, False)

        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(self, idx, clip_feature, key_padding_mask=None):
        # key_padding_mask (here named key_padding_mask_for_idxs for clarity within this function)
        # is for `idx` (shape B, T_idx, dim). True for padded positions in idx, especially in the semantic part.
        key_padding_mask_for_idxs = key_padding_mask 

        condition_embedding = self.cond_emb(clip_feature).unsqueeze(1) # (B, 1, C)
        mask_for_blocks = None

        if len(idx) == 0:
            token_embeddings = condition_embedding
            # Mask for blocks is just for the condition embedding (i.e., False, do not mask)
            mask_for_blocks = torch.zeros(condition_embedding.shape[0], 1, dtype=torch.bool, device=condition_embedding.device)
        else:
            b, t_idx, dim = idx.size()
            assert t_idx <= self.block_size, f"Cannot forward, idx sequence length {t_idx} exceeds model block size {self.block_size}."

            # Prepare token embeddings from idx based on whether it's semantic, reconstruction, or both
            # if t_idx <= self.semantic_len : # Only semantic tokens (or shorter than self.semantic_len)
            #     token_embeddings_unconditioned = self.tok_emb[0](idx)
            # else: # Both semantic and reconstruction tokens
            #     token_sem_embeddings = self.tok_emb[0](idx[..., :self.semantic_len])
            #     token_recon_embeddings = self.tok_emb[1](idx[..., self.semantic_len:t_idx]) # Slice up to t_idx
            #     token_embeddings_unconditioned = torch.cat([token_sem_embeddings, token_recon_embeddings], dim=1)
            token_embeddings_unconditioned = idx
            token_embeddings = torch.cat([condition_embedding, token_embeddings_unconditioned], dim=1)
            
            # Adjust key_padding_mask_for_idxs for the prepended condition embedding
            if key_padding_mask_for_idxs is not None:
                # Ensure key_padding_mask_for_idxs matches the length of idx used for embeddings
                key_padding_mask_for_idxs = key_padding_mask_for_idxs[:, :t_idx + 1]
                # mask_for_blocks = torch.cat([
                #     # torch.zeros(b, 1, dtype=torch.bool, device=idx.device), # False for condition
                #     key_padding_mask_for_idxs[:, :t_idx] # Use mask corresponding to actual idx length
                # ], dim=1)
            else:
                # If no original mask, then no padding for any part of the sequence passed to blocks (beyond causal)
                # Create a mask of all Falses for the blocks if none provided.
                mask_for_blocks = torch.zeros(token_embeddings.shape[0], token_embeddings.shape[1], dtype=torch.bool, device=idx.device)
            
        x = self.pos_embed(token_embeddings)
        
        for block in self.blocks:
            x = block(x, key_padding_mask=mask_for_blocks) # Pass the adjusted mask
            
        return x


class CrossCondTransHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4):
        super().__init__()

        self.blocks = nn.Sequential(*[Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x):
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    
class CrossCondTransDualHead(nn.Module):

    def __init__(self, 
                num_vq=1024, 
                embed_dim=512, 
                block_size=16, 
                num_layers=2, 
                n_head=8, 
                drop_out_rate=0.1, 
                fc_rate=4,
                semantic_len = 50):
        super().__init__()

        self.blocks = nn.ModuleList([Block(embed_dim, block_size, n_head, drop_out_rate, fc_rate) for _ in range(num_layers)])
        self.ln_f = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(2)])
        self.sem_heads = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.recon_heads = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.semantic_len = semantic_len
        self.num_vq = num_vq
        # self.ln_f = nn.LayerNorm(embed_dim)
        # self.head = nn.Linear(embed_dim, num_vq + 1, bias=False)
        self.block_size = block_size

        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(self, x, key_padding_mask=None):
        # x is the output from TransBase (Dual), includes condition embedding.
        # key_padding_mask is also for x (i.e., includes a False for the condition part at the beginning).
        # We call this key_padding_mask_for_input_x for clarity within this function.
        key_padding_mask_for_input_x = key_padding_mask

        for block in self.blocks:
            x = block(x, key_padding_mask=key_padding_mask_for_input_x) # Pass the mask to each block

        # The input x has shape (B, 1 + T_idx, C) due to condition embedding.
        # T_idx is the original length of idx tokens (semantic + reconstruction).
        # self.semantic_len is the length of semantic part in *original idxs*.
        
        # Slicing for semantic and reconstruction parts after blocks:
        # Semantic part in x (after condition): x[:, 1 : 1 + self.semantic_len, :]
        # Reconstruction part in x (after condition and semantic): x[:, 1 + self.semantic_len :, :]

        # Note: The slicing here assumes x's length is at least 1 + self.semantic_len for the semantic part,
        # and potentially longer for the reconstruction part.
        # If x is shorter, e.g., during early generation steps, these slices might be empty or out of bounds.
        # The lengths should be handled carefully based on actual sequence length of x.
        # current_seq_len_after_cond = x.shape[1]

        # Determine the end index for the semantic part in x
        # It's 1 (for condition) + length of semantic tokens in x
        # Max length of semantic tokens in x is self.semantic_len
        # Actual length of semantic tokens in x is min(self.semantic_len, current_seq_len_after_cond)
        # semantic_part_end_in_x = 1 + min(self.semantic_len, current_seq_len_after_cond)
        
        x_semantic_part = x[:, :self.semantic_len, :]
        x_recon_part = x[:, self.semantic_len:, :] # Takes the rest
        
        logits_semantic = torch.empty(x.shape[0], 0, self.sem_heads.out_features, device=x.device)
        if x_semantic_part.shape[1] > 0:
            x_semantic_out = self.ln_f[0](x_semantic_part)
            logits_semantic = self.sem_heads(x_semantic_out)
        
        logits_recon = torch.empty(x.shape[0], 0, self.recon_heads.out_features, device=x.device)
        if x_recon_part.shape[1] > 0:
            x_recon_out = self.ln_f[1](x_recon_part)
            logits_recon = self.recon_heads(x_recon_out)
        
        # The logits should correspond to the positions *after* the condition token.
        logits_result = torch.cat([logits_semantic, logits_recon], dim=1)
        return logits_result

        
