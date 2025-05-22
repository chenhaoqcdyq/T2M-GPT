import torch.nn as nn
from models.quantize_cnn import QuantizeEMAReset
from models.resnet import Resnet1D
import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from transformers import BertTokenizer, BertModel
from models.vq.residual_vq import ResidualVQ

class Encoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None,
                 causal=False):
        super().__init__()
        
        blocks = []
        if stride_t == 1:
            pad_t, filter_t = 1, 3
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2
        # filter_t, pad_t = stride_t * 2, stride_t // 2
        if causal:
            # blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
            blocks.append(nn.ConstantPad1d((2,0), 0))  # kernel_size=3的因果填充
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        if down_t < 3:
            for i in range(down_t):
                input_dim = width
                if causal:
                    causal_pad = (filter_t-1)
                    block = nn.Sequential(
                        nn.ConstantPad1d((causal_pad,0), 0),  # 左侧填充
                        nn.Conv1d(input_dim, width, filter_t, stride_t, 0),
                        Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal),
                    )
                else:
                    block = nn.Sequential(
                        nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                        Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
                    )
                blocks.append(block)
        else:
            for i in range(2):
                input_dim = width
                if causal:
                    causal_pad = (filter_t-1)
                    block = nn.Sequential(
                        nn.ConstantPad1d((causal_pad,0), 0),  # 左侧填充
                        nn.Conv1d(input_dim, width, filter_t, stride_t, 0),
                        Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal),
                    )
                else:
                    block = nn.Sequential(
                        nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                        Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
                    )
                blocks.append(block)
            if causal:
                causal_pad = (filter_t-1)
                block = nn.Sequential(
                        nn.ConstantPad1d((causal_pad,0), 0),  # 左侧填充
                        nn.Conv1d(input_dim, width, filter_t, stride_t, 0),
                    )
            else:
                block = nn.Sequential(
                    nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                )
            blocks.append(block)
        if causal:
            blocks.append(nn.ConstantPad1d((2,0), 0))  # kernel_size=3
            blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, motion_mask = None):
        if motion_mask is not None:
            x = x * motion_mask.unsqueeze(1)
        return self.model(x)

class Encoder_causal(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3,
                 activation='relu',
                 norm=None):
        super().__init__()
        
        blocks = []
        if stride_t == 1:
            pad_t, filter_t = 1, 3
        else:
            filter_t, pad_t = stride_t * 2, stride_t // 2
        # filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        
        for i in range(down_t):
            input_dim = width
            block = nn.Sequential(
                nn.Conv1d(input_dim, width, filter_t, stride_t, pad_t),
                Resnet1D(width, depth, dilation_growth_rate, activation=activation, norm=norm),
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, output_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x, motion_mask = None):
        if motion_mask is not None:
            x = x * motion_mask.unsqueeze(1)
        return self.model(x)

class CausalTransformerEncoder(nn.TransformerEncoder):
    """带因果掩码的Transformer编码器"""
    def forward(self, src, mask=None, **kwargs):
        # 自动生成因果掩码
        if mask is None:
            device = src.device
            seq_len = src.size(1)
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
            return super().forward(src, mask=causal_mask, **kwargs)
        return super().forward(src, mask, **kwargs)

class DynamicProjection(nn.Module):
    """动态特征投影"""
    def __init__(self, in_dim, d_model):
        super().__init__()
        self.base_proj = nn.Sequential(
            nn.Conv1d(in_dim, d_model//2, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model//2, d_model, 3, padding=1)
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(d_model, d_model, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        base = self.base_proj(x)
        gate = self.gate(base)
        return base * gate

class Encoder_Transformer(nn.Module):
    def __init__(self,
                 dim = 251,
                 d_model=512,
                 nhead = 8,
                 num_layers = 3,
                 down_sample = False):
        super().__init__()
        self.part_projs = DynamicProjection(dim, d_model=d_model)
        time_encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4*d_model,
            batch_first=True
        )
        self.down_sample_flag = down_sample
        if self.down_sample_flag:
            self.down_sample = CausalDownsample(d_model, 2)
        else:
            self.down_sample = nn.Identity()
        # self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.time_transformer = CausalTransformerEncoder(time_encoder_layer, num_layers=num_layers)
    
    def forward(self, motion_feature, motion_mask=None):
        # 部件特征预处理
        B, T = motion_feature.shape[0], motion_feature.shape[1]
        motion_feature = self.part_projs(motion_feature).permute(0,2,1)
        if motion_mask is not None:
            motion_mask = motion_mask.to(motion_feature.device).bool()
            if self.down_sample_flag:
                motion_feature = self.down_sample(motion_feature, padding_mask=~motion_mask)
                if torch.isnan(motion_feature).any():
                    print(motion_feature)
                motion_mask = motion_mask[:, ::4].clone()
            time_feat = self.time_transformer(motion_feature, src_key_padding_mask=~motion_mask)  # [T, B*7, d]
            if torch.isnan(time_feat).any():
                print(time_feat)
        else:
            if self.down_sample_flag:
                motion_feature = self.down_sample(motion_feature)
            time_feat = self.time_transformer(motion_feature)
        # 残差连接增强
        return time_feat.permute(0,2,1)

class Decoder(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None,
                 causal=False):
        super().__init__()
        blocks = []
        self.causal = causal
        filter_t, pad_t = stride_t * 2, stride_t // 2
        if causal:
            # blocks.append(nn.Conv1d(input_emb_width, width, 3, 1, 1))
            blocks.append(nn.ConstantPad1d((2,0), 0))  # kernel_size=3的因果填充
            blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        if down_t < 3:
            for i in range(down_t):
                out_dim = width
                if causal:
                    block = nn.Sequential(
                        Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, causal=causal),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.ConstantPad1d((2,0), 0),
                        nn.Conv1d(width, out_dim, 3, 1, 0)
                    )
                else:  
                    block = nn.Sequential(
                        Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv1d(width, out_dim, 3, 1, 1)
                    )
                blocks.append(block)
        else:
            for i in range(2):
                out_dim = width
                if causal:
                    block = nn.Sequential(
                        Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm, causal=causal),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.ConstantPad1d((2,0), 0),
                        nn.Conv1d(width, out_dim, 3, 1, 0)
                    )
                else:  
                    block = nn.Sequential(
                        Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv1d(width, out_dim, 3, 1, 1)
                    )
                blocks.append(block)
            block = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        if causal:
            blocks.append(nn.ConstantPad1d((2,0), 0))
            blocks.append(nn.Conv1d(width, width, 3, 1, 0))
            blocks.append(nn.ReLU())
            blocks.append(nn.ConstantPad1d((2,0), 0))
            blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 0))
        else:
            blocks.append(nn.Conv1d(width, width, 3, 1, 1))
            blocks.append(nn.ReLU())
            blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)
    
class Decoder_wo_upsample(nn.Module):
    def __init__(self,
                 input_emb_width = 3,
                 output_emb_width = 512,
                 down_t = 3,
                 stride_t = 2,
                 width = 512,
                 depth = 3,
                 dilation_growth_rate = 3, 
                 activation='relu',
                 norm=None):
        super().__init__()
        blocks = []
        
        filter_t, pad_t = stride_t * 2, stride_t // 2
        blocks.append(nn.Conv1d(output_emb_width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        for i in range(down_t):
            out_dim = width
            block = nn.Sequential(
                Resnet1D(width, depth, dilation_growth_rate, reverse_dilation=True, activation=activation, norm=norm),
                # nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv1d(width, out_dim, 3, 1, 1)
            )
            blocks.append(block)
        blocks.append(nn.Conv1d(width, width, 3, 1, 1))
        blocks.append(nn.ReLU())
        blocks.append(nn.Conv1d(width, input_emb_width, 3, 1, 1))
        self.model = nn.Sequential(*blocks)

    def forward(self, x):
        return self.model(x)

class LGVQ(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=4,  # 减少注意力头数
                 num_layers=2,  # 减少Transformer层数
                 bert_hidden_dim=768,
                 vocab_size=30522,
                 dropout=0.2,   # 增加dropout率
                 down_sample = 0,
                 causal = True,
                 layer_norm = False):  
        super().__init__()
        self.vocab_size = vocab_size
        self.args = args
        # 添加时间降采样层
        self.ifdown_sample = down_sample
        if down_sample==1:
            self.time_downsamplers = nn.ModuleList([
                TemporalDownsamplerHalf(d_model, causal=causal, layer_norm=layer_norm) for _ in range(num_layers)
            ])
        elif down_sample==0:
            self.time_downsamplers = nn.ModuleList([
                nn.Identity() for _ in range(num_layers)
            ])
        else:
            self.time_downsamplers = CausalDownsample(d_model, down_sample)
            
        self.time_position = nn.Parameter(torch.randn(1, 196, d_model))  # 时间步编码(假设最大序列长度196)
        self.time_transformer = CausalTransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2*d_model,  # 减少FFN维度
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )

        # BERT
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False  # 默认冻结

        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.text_motion_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        
        # 跨模态注意力
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_hidden_dim,
                nhead=4,  # 减少注意力头
                dim_feedforward=2*bert_hidden_dim,  # 降低FFN维度
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)  # 减少层数
        ])

        # 对比学习
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        
        # 数据增强
        self.motion_aug = ComposeAugmentation([
            # TemporalCrop(max_ratio=0.2),
            FeatureJitter(std=0.0)
        ])
        
        # 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
        )
        
        # MLM head with label smoothing
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        
        # 运动文本投影
        self.motion_text_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        # if args.num_quantizers > 1:
        #     rvqvae_config = {
        #         'num_quantizers': args.num_quantizers,
        #         'shared_codebook': False,
        #         'quantize_dropout_prob': 0.2,
        #         'quantize_dropout_cutoff_index': 0,
        #         'nb_code': args.nb_code,
        #         'code_dim': d_model, 
        #         'args': args,
        #     }
        #     self.sem_quantizer = ResidualVQ(**rvqvae_config)
        # else:
        #     self.sem_quantizer = QuantizeEMAReset(args.nb_code, d_model, args)
    
    def encode(self, motion, motion_mask=None):
        B, T = motion.shape[0], motion.shape[1]
        # 时间特征处理
        time_feat = motion
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            # time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat, padding_mask=~motion_mask)
                if self.ifdown_sample:
                    motion_mask = motion_mask[:, ::2]  # 更新mask
                time_feat = layer(time_feat, src_key_padding_mask=~motion_mask)
        else:
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                time_feat = layer(time_feat)
            
        # 特征重组
        feature = time_feat
        sem_idx = self.sem_quantizer.quantize(feature)
        return sem_idx
        
    def text_motion_topk(self, motion, text, text_mask=None, motion_mask=None, topk=5):
        B, T = motion.shape[0], motion.shape[1]
        # 时间特征处理
        time_feat = motion
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            if self.ifdown_sample == 2:
                time_feat = self.time_downsamplers(time_feat, padding_mask=~motion_mask)
                motion_mask = motion_mask[:, ::4].clone()
                time_feat = self.time_transformer(time_feat, src_key_padding_mask=~motion_mask)
            else:
                # 在每一层Transformer后应用时间降采样
                for i, layer in enumerate(self.time_transformer.layers):
                    time_feat = self.time_downsamplers[i](time_feat)
                    if self.ifdown_sample:
                        # motion_mask = motion_mask[:, ::2]  # 更新mask
                        motion_mask = motion_mask[:, ::2].clone()  # 使用 clone() 创建副本
                    time_feat = layer(time_feat, src_key_padding_mask=~motion_mask)
        else:
            if self.ifdown_sample == 2:
                time_feat = self.time_downsamplers(time_feat)
                time_feat = self.time_transformer(time_feat)
            else:
                # 在每一层Transformer后应用时间降采样
                for i, layer in enumerate(self.time_transformer.layers):
                    time_feat = self.time_downsamplers[i](time_feat)
                    time_feat = layer(time_feat)
            
        # 特征重组
        cls_token = time_feat
        # if self.args.num_quantizers > 1:
        #     cls_token, all_index, loss_commit, perplexity = self.sem_quantizer(feature.permute(0,2,1), sample_codebook_temp=0.5)
        # else:
        #     cls_token, loss_commit, perplexity = self.sem_quantizer(feature.permute(0,2,1))
        # cls_token = cls_token.permute(0,2,1)
        global_feat = cls_token.mean(dim=1)
        
        if text_mask is not None:
            # text_feature, text_id = text
            # if text_mask is not None:
            input_ids = text_mask['input_ids'].to(motion.device)
            labels_text = text_mask['labels'].to(motion.device).float()
            attention_mask = text_mask['attention_mask'].to(motion.device).bool()
            
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion.device).float()
                text_feature_pooler = text_mask['feature'].to(motion.device).float()
                    
            # 特征投影
            text_feature = text_feature.to(motion.device).float()
            text_query = self.text_proj(text_feature)
            
            motion_query = self.motion_all_proj(cls_token)
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
            
            # # 文本特征提取
            bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            # encoded = bert_tokenizer(
            #     text,
            #     padding='max_length',
            #     truncation=True,
            #     max_length=128,
            #     return_tensors='pt'
            # )
            # for k, v in encoded.items():
            #     encoded[k] = v.to(motion[0].device)
            # bert_outputs = self.bert_model(**encoded)
            # text_feature = bert_outputs.pooler_output.to(motion[0].device).float()
            text_feature_pooler = self.text_motion_proj(text_feature_pooler)
            
            motion_feature_global = self.motion_text_proj(global_feat)
            
            # 计算相似度矩阵
            motion_feature_global = F.normalize(motion_feature_global, p=2, dim=-1)  # [B, d]
            text_feature_pooler = F.normalize(text_feature_pooler, p=2, dim=-1)       # [B, d]
            similarity_matrix = torch.mm(motion_feature_global, text_feature_pooler.T)
            
            # 计算召回指标
            batch_size = similarity_matrix.size(0)
            labels = torch.arange(batch_size).to(similarity_matrix.device)  # 对角线是正确匹配
            
            # 计算Top-K匹配
            _, topk_indices = similarity_matrix.topk(topk, dim=1)  # [B, K]
            
            # 统计各召回率
            correct_r1 = (topk_indices[:, 0] == labels).float().sum().cpu().item()
            correct_r3 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()
            correct_r5 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()

            # 可视化文本匹配结果（仅在Top-1不匹配时输出）
            if batch_size <= 16:  # 只在小batch size时可视化
                for i in range(batch_size):
                    # if topk_indices[i, 0] != labels[i]:  # 只在Top-1不匹配时输出
                    #     print("\n=== 文本匹配可视化（Top-1不匹配） ===")
                    print(f"\n样本 {i+1}:")
                    print(f"真值文本: {text[i]}")
                    print(f"Top-{topk} 匹配文本:")
                    for j in range(topk):
                        matched_idx = topk_indices[i, j].item()
                        similarity = similarity_matrix[i, matched_idx].item()
                        print(f"  {j+1}. {text[matched_idx]} (相似度: {similarity:.4f})")
            
            # MLM预测
            logits = self.mlm_head(text_query)
            # 计算MLM任务的Top-K召回率
            active_loss = (labels_text != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels_text.view(-1)[active_loss]
            
            topk_values, topk_indices = active_logits.topk(k=5, dim=-1)  # [active_num, 5]
            active_labels = active_labels.long()  # [active_num]
            expanded_labels = active_labels.unsqueeze(1).expand(-1, 5)  # [active_num, 5]
            hits = (topk_indices == expanded_labels)  # [active_num, 5]
            
            if batch_size <= 16:  # 只在小batch size时可视化
                num_tmp = 0
                num_tmp_list = torch.zeros(batch_size, hits.shape[0], dtype=torch.long, device=motion[0].device)
                for i in range(batch_size):
                    # 获取原始文本
                    original_text = bert_tokenizer.decode(input_ids[i], skip_special_tokens=False)
                    # 获取掩码位置
                    mask_positions = (labels[i] != -100).nonzero().squeeze(-1)
                    
                    if len(mask_positions) == 0:  # 如果没有掩码位置，跳过
                        continue
                    # 检查是否有Top-1预测错误的位置
                    has_error = False
                    for pos in mask_positions:
                        num_tmp += 1
                        num_tmp_list[i][num_tmp - 1] = pos
                        if not hits[num_tmp-1][0]:  # 检查Top-1是否正确
                            has_error = True
                            # break
                    # 只在有预测错误时输出
                    if has_error:
                        print("\n=== MLM预测可视化（Top-1预测错误） ===")
                        print(f"\n样本 {i+1}:")
                        print(f"原始文本: {text[i]}")
                        print(f"mask文本: {original_text.replace(' [PAD]','')}")
                        print("掩码位置预测:")
                        for j, pos in enumerate(mask_positions):
                            # if pos < hits.shape[0]:  # 确保位置有效
                            index = torch.where(num_tmp_list[i] == pos)[0].item()
                            top_preds = topk_indices[index]  # [5]
                            pred_tokens = [bert_tokenizer.decode([idx]) for idx in top_preds]
                            gt_token = bert_tokenizer.decode([active_labels[index]])
                            print(f"  位置 {pos}:")
                            print(f"    真实token: {gt_token}")
                            print(f"    预测token: {pred_tokens}")
            
            r1_mlm = hits[:, 0].sum().float() / active_labels.size(0)
            r3_mlm = hits[:, :3].sum(dim=1).clamp(max=1).sum().float() / active_labels.size(0)
            r5_mlm = hits.sum().float() / active_labels.size(0)
            
            return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], \
                   [r1_mlm.cpu().item(), r3_mlm.cpu().item(), r5_mlm.cpu().item()]

    def forward(self, motion, text_mask=None, motion_mask=None, text_id=None):
        # 部件特征预处理 bs,6,seq,d
        B, T = motion.shape[0], motion.shape[1]
    
        # 数据增强
        if self.training:
            motion = self.motion_aug(motion)
            
        # 时间特征处理
        time_feat = motion
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            if self.ifdown_sample == 2:                
                time_feat = self.time_downsamplers(time_feat, padding_mask=~motion_mask)
                motion_mask = motion_mask[:, ::4].clone()
                time_feat = self.time_position[:, :time_feat.shape[1], :] + time_feat
                time_feat = self.time_transformer(time_feat, src_key_padding_mask=~motion_mask)
            else:
                # 在每一层Transformer后应用时间降采样
                time_feat = self.time_position[:, :time_feat.shape[1], :] + time_feat
                for i, layer in enumerate(self.time_transformer.layers):
                    time_feat = self.time_downsamplers[i](time_feat)
                    if self.ifdown_sample:
                        motion_mask = motion_mask[:, ::2].clone()  # 使用 clone() 创建副本
                    time_feat = layer(time_feat, src_key_padding_mask=~motion_mask)
        else:
            if self.ifdown_sample == 2:
                time_feat = self.time_downsamplers(time_feat)
                time_feat = self.time_position[:, :time_feat.shape[1], :] + time_feat
                time_feat = self.time_transformer(time_feat)
            else:
                # 在每一层Transformer后应用时间降采样
                time_feat = self.time_position[:, :time_feat.shape[1], :] + time_feat
                for i, layer in enumerate(self.time_transformer.layers):
                    time_feat = self.time_downsamplers[i](time_feat)
                    time_feat = layer(time_feat)
            
        # 特征重组
        cls_token = time_feat
        # if self.args.num_quantizers > 1:
        #     cls_token, all_index, loss_commit, perplexity = self.sem_quantizer(feature.permute(0,2,1), sample_codebook_temp=0.5)
        # else:
        #     cls_token, loss_commit, perplexity = self.sem_quantizer(feature.permute(0,2,1))
        # cls_token = cls_token.permute(0,2,1)
        global_feat = cls_token.mean(dim=1)
        loss_commit, perplexity = torch.tensor(0.0).to(motion.device), torch.tensor(0.0).to(motion.device)
        if text_mask is not None:
            # text_feature, text_id = text
            # if text_mask is not None:
            input_ids = text_mask['input_ids'].to(motion.device)
            labels = text_mask['labels'].to(motion.device).float()
            attention_mask = text_mask['attention_mask'].to(motion.device).bool()
            
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion.device).float()
                text_feature_pooler = text_mask['feature'].to(motion.device).float()
                    
            # 特征投影
            text_feature = text_feature.to(motion.device).float()
            text_query = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)
            motion_query = self.motion_all_proj(cls_token)
            # if self.ifdown_sample:
            #     motion_mask = motion_mask[:, ::4]
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 标签平滑的MLM损失
            # loss_fct = LabelSmoothingCrossEntropy(smoothing=0.05)
            # active_loss = (labels != -100).view(-1)
            # active_logits = logits.view(-1, self.vocab_size)[active_loss]
            # active_labels = labels.view(-1)[active_loss]
            # mlm_loss = loss_fct(active_logits, active_labels.long())
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            mlm_loss = loss_fct(active_logits, active_labels.long())
            # 对比损失
            text_feature_pooler = self.text_motion_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_pooler, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(motion.device)
            mlm_loss = torch.tensor(0.0).to(motion.device)
        
        return cls_token, [contrastive_loss, mlm_loss], [loss_commit, perplexity]
        
        
class Dualsem_encoderv3(nn.Module):
    def __init__(self, args,
                 d_model=256,
                 nhead=4,  # 减少注意力头数
                 num_layers=2,  # 减少Transformer层数
                 bert_hidden_dim=768,
                 vocab_size=30522,
                 dropout=0.2,   # 增加dropout率
                 down_sample = 0,
                 causal = True):  
        super().__init__()
        self.vocab_size = vocab_size
        self.args = args
        # 添加时间降采样层
        self.ifdown_sample = down_sample
        if down_sample==1:
            self.time_downsamplers = nn.ModuleList([
                TemporalDownsamplerHalf(d_model, causal=causal) for _ in range(num_layers)
            ])
        elif down_sample==0:
            self.time_downsamplers = nn.ModuleList([
                nn.Identity() for _ in range(num_layers)
            ])
        else:
            self.time_downsamplers = CausalDownsample(d_model, down_sample)
            
        self.time_position = nn.Parameter(torch.randn(1, 196, d_model))  # 时间步编码(假设最大序列长度196)
        self.time_transformer = CausalTransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=2*d_model,  # 减少FFN维度
                dropout=dropout,
                batch_first=True
            ),
            num_layers=num_layers
        )
        self.time_position = nn.Parameter(torch.randn(1, 196, d_model))

        # BERT
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        for param in self.bert_model.parameters():
            param.requires_grad = False  # 默认冻结

        self.text_proj = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        self.text_motion_proj = nn.Linear(bert_hidden_dim, bert_hidden_dim)
        
        # 跨模态注意力
        self.cross_attn_layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=bert_hidden_dim,
                nhead=4,  # 减少注意力头
                dim_feedforward=2*bert_hidden_dim,  # 降低FFN维度
                dropout=dropout,
                batch_first=True
            ) for _ in range(2)  # 减少层数
        ])

        # 对比学习
        self.contrastive_loss = ContrastiveLossWithSTSV2()
        
        # 数据增强
        self.motion_aug = ComposeAugmentation([
            # TemporalCrop(max_ratio=0.2),
            FeatureJitter(std=0.0)
        ])
        
        # 运动特征处理模块
        self.motion_all_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
        )
        
        # MLM head with label smoothing
        self.mlm_head = nn.Sequential(
            nn.Linear(bert_hidden_dim, bert_hidden_dim * 4),
            nn.GELU(),
            nn.LayerNorm(bert_hidden_dim * 4),
            nn.Dropout(dropout),
            nn.Linear(bert_hidden_dim * 4, vocab_size)
        )
        
        # 运动文本投影
        self.motion_text_proj = nn.Sequential(
            nn.Linear(d_model, bert_hidden_dim),
            nn.LayerNorm(bert_hidden_dim),
            nn.Dropout(dropout),
            nn.GELU()
        )
        if args.num_quantizers > 1:
            rvqvae_config = {
                'num_quantizers': args.num_quantizers,
                'shared_codebook': False,
                'quantize_dropout_prob': 0.2,
                'quantize_dropout_cutoff_index': 0,
                'nb_code': args.nb_code,
                'code_dim': d_model, 
                'args': args,
            }
            self.sem_quantizer = ResidualVQ(**rvqvae_config)
        else:
            self.sem_quantizer = QuantizeEMAReset(args.nb_code, d_model, args)
    
    def encode(self, motion, motion_mask=None):
        B, T = motion.shape[0], motion.shape[1]
        # 时间特征处理
        time_feat = motion
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            # time_key_padding_mask = motion_mask.repeat_interleave(7, dim=0)
            
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat, padding_mask=~motion_mask)
                if self.ifdown_sample:
                    motion_mask = motion_mask[:, ::2]  # 更新mask
                time_feat = layer(time_feat, src_key_padding_mask=~motion_mask)
        else:
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                time_feat = layer(time_feat)
            
        # 特征重组
        feature = time_feat
        if self.args.num_quantizers > 1:
            sem_idx = self.sem_quantizer.quantize(feature.permute(0,2,1))
        else:
            sem_idx = self.sem_quantizer.quantize(feature)
        return sem_idx
    
    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        """
        计算动作和文本之间的Top-K匹配
        Args:
            motion: 动作特征列表 [6, B, T, D]
            text: 文本字符串
            motion_mask: 动作掩码 [B, T]
            topk: 返回的top-k结果数
            text_mask: 文本掩码字典
        Returns:
            [r1, r3, r5]: 召回率指标
            [r1_mlm, r3_mlm, r5_mlm]: MLM任务的召回率指标
        """
        # 部件特征预处理
        B, T = motion.shape[0], motion.shape[1]
    
        # 数据增强
        if self.training:
            motion = self.motion_aug(motion)
            
        # 时间特征处理
        time_feat = motion
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                if self.ifdown_sample:
                    # motion_mask = motion_mask[:, ::2]  # 更新mask
                    motion_mask = motion_mask[:, ::2].clone()  # 使用 clone() 创建副本
                time_feat = layer(time_feat, src_key_padding_mask=~motion_mask)
        else:
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                time_feat = layer(time_feat)
            
        # 特征重组
        feature = time_feat
        if self.args.num_quantizers > 1:
            cls_token, all_index, loss_commit, perplexity = self.sem_quantizer(feature.permute(0,2,1), sample_codebook_temp=0.5)
        else:
            cls_token, loss_commit, perplexity = self.sem_quantizer(feature.permute(0,2,1))
        cls_token = cls_token.permute(0,2,1)
        global_feat = cls_token.mean(dim=1)
        motion_feature_global = self.motion_text_proj(global_feat)

          # 文本特征提取
        bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        encoded = bert_tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        for k, v in encoded.items():
            encoded[k] = v.to(motion[0].device)
        bert_outputs = self.bert_model(**encoded)
        text_feature = bert_outputs.pooler_output.to(motion[0].device).float()
        text_feature_pooler = self.text_motion_proj(text_feature)
        
        # 计算相似度矩阵
        motion_feature_global = F.normalize(motion_feature_global, p=2, dim=-1)  # [B, d]
        text_feature_pooler = F.normalize(text_feature_pooler, p=2, dim=-1)       # [B, d]
        similarity_matrix = torch.mm(motion_feature_global, text_feature_pooler.T)
        
        # 计算召回指标
        batch_size = similarity_matrix.size(0)
        labels = torch.arange(batch_size).to(similarity_matrix.device)  # 对角线是正确匹配
        
        # 计算Top-K匹配
        _, topk_indices = similarity_matrix.topk(topk, dim=1)  # [B, K]
        
        # 统计各召回率
        correct_r1 = (topk_indices[:, 0] == labels).float().sum().cpu().item()
        correct_r3 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()
        correct_r5 = (topk_indices == labels.unsqueeze(1)).any(dim=1).float().sum().cpu().item()

        # 可视化文本匹配结果（仅在Top-1不匹配时输出）
        if batch_size <= 16:  # 只在小batch size时可视化
            for i in range(batch_size):
                # if topk_indices[i, 0] != labels[i]:  # 只在Top-1不匹配时输出
                #     print("\n=== 文本匹配可视化（Top-1不匹配） ===")
                print(f"\n样本 {i+1}:")
                print(f"真值文本: {text[i]}")
                print(f"Top-{topk} 匹配文本:")
                for j in range(topk):
                    matched_idx = topk_indices[i, j].item()
                    similarity = similarity_matrix[i, matched_idx].item()
                    print(f"  {j+1}. {text[matched_idx]} (相似度: {similarity:.4f})")

        
        if text_mask is not None:
            # text_feature, text_id = text
            # if text_mask is not None:
            input_ids = text_mask['input_ids'].to(motion.device)
            labels = text_mask['labels'].to(motion.device).float()
            attention_mask = text_mask['attention_mask'].to(motion.device).bool()
            
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion.device).float()
                text_feature_pooler = text_mask['feature'].to(motion.device).float()
                    
            # 特征投影
            text_feature = text_feature.to(motion.device).float()
            text_query = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)
            motion_query = self.motion_all_proj(cls_token)
            # if self.ifdown_sample:
            #     motion_mask = motion_mask[:, ::4]
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 标签平滑的MLM损失
            loss_fct = LabelSmoothingCrossEntropy(smoothing=0.05)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            mlm_loss = loss_fct(active_logits, active_labels.long())
            # loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            # active_loss = (labels != -100).view(-1)
            # active_logits = logits.view(-1, self.vocab_size)[active_loss]
            # active_labels = labels.view(-1)[active_loss]
            # mlm_loss = loss_fct(active_logits, active_labels.long())
            # 对比损失
            topk_values, topk_indices = active_logits.topk(k=5, dim=-1)  # [active_num, 5]
            active_labels = active_labels.long()  # [active_num]
            expanded_labels = active_labels.unsqueeze(1).expand(-1, 5)  # [active_num, 5]
            hits = (topk_indices == expanded_labels)  # [active_num, 5]
            
            if batch_size <= 16:  # 只在小batch size时可视化
                num_tmp = 0
                num_tmp_list = torch.zeros(batch_size, hits.shape[0], dtype=torch.long, device=motion[0].device)
                for i in range(batch_size):
                    # 获取原始文本
                    original_text = bert_tokenizer.decode(input_ids[i], skip_special_tokens=False)
                    # 获取掩码位置
                    mask_positions = (labels[i] != -100).nonzero().squeeze(-1)
                    
                    if len(mask_positions) == 0:  # 如果没有掩码位置，跳过
                        continue
                    # 检查是否有Top-1预测错误的位置
                    has_error = False
                    for pos in mask_positions:
                        num_tmp += 1
                        num_tmp_list[i][num_tmp - 1] = pos
                        if not hits[num_tmp-1][0]:  # 检查Top-1是否正确
                            has_error = True
                            # break
                    # 只在有预测错误时输出
                    if has_error:
                        print("\n=== MLM预测可视化（Top-1预测错误） ===")
                        print(f"\n样本 {i+1}:")
                        print(f"原始文本: {text[i]}")
                        print(f"mask文本: {original_text.replace(' [PAD]','')}")
                        print("掩码位置预测:")
                        for j, pos in enumerate(mask_positions):
                            # if pos < hits.shape[0]:  # 确保位置有效
                            index = torch.where(num_tmp_list[i] == pos)[0].item()
                            top_preds = topk_indices[index]  # [5]
                            pred_tokens = [bert_tokenizer.decode([idx]) for idx in top_preds]
                            gt_token = bert_tokenizer.decode([active_labels[index]])
                            print(f"  位置 {pos}:")
                            print(f"    真实token: {gt_token}")
                            print(f"    预测token: {pred_tokens}")
            
            r1_mlm = hits[:, 0].sum().float() / active_labels.size(0)
            r3_mlm = hits[:, :3].sum(dim=1).clamp(max=1).sum().float() / active_labels.size(0)
            r5_mlm = hits.sum().float() / active_labels.size(0)
            
            return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], \
                   [r1_mlm.cpu().item(), r3_mlm.cpu().item(), r5_mlm.cpu().item()]
                   
        return [correct_r1/batch_size, correct_r3/batch_size, correct_r5/batch_size], [0, 0, 0]
   

    def forward(self, motion, text_mask=None, motion_mask=None, text_id=None):
        # 部件特征预处理 bs,6,seq,d
        B, T = motion.shape[0], motion.shape[1]
    
        # 数据增强
        if self.training:
            motion = self.motion_aug(motion)
            
        # 时间特征处理
        time_feat = motion
        if motion_mask is not None:
            motion_mask = motion_mask.to(time_feat.device).bool()
            time_feat = self.time_position[:,:time_feat.shape[1],:]+time_feat
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                if self.ifdown_sample:
                    # motion_mask = motion_mask[:, ::2]  # 更新mask
                    motion_mask = motion_mask[:, ::2].clone()  # 使用 clone() 创建副本
                time_feat = layer(time_feat, src_key_padding_mask=~motion_mask)
        else:
            # 在每一层Transformer后应用时间降采样
            for i, layer in enumerate(self.time_transformer.layers):
                time_feat = self.time_downsamplers[i](time_feat)
                time_feat = layer(time_feat)
            
        # 特征重组
        feature = time_feat
        if self.args.num_quantizers > 1:
            cls_token, all_index, loss_commit, perplexity = self.sem_quantizer(feature.permute(0,2,1), sample_codebook_temp=0.5)
        else:
            cls_token, loss_commit, perplexity = self.sem_quantizer(feature.permute(0,2,1))
        cls_token = cls_token.permute(0,2,1)
        global_feat = cls_token.mean(dim=1)
        
        if text_mask is not None:
            # text_feature, text_id = text
            # if text_mask is not None:
            input_ids = text_mask['input_ids'].to(motion.device)
            labels = text_mask['labels'].to(motion.device).float()
            attention_mask = text_mask['attention_mask'].to(motion.device).bool()
            
            with torch.no_grad():
                bert_outputs = self.bert_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                text_feature = bert_outputs.last_hidden_state.to(motion.device).float()
                text_feature_pooler = text_mask['feature'].to(motion.device).float()
                    
            # 特征投影
            text_feature = text_feature.to(motion.device).float()
            text_query = self.text_proj(text_feature)
            motion_feature_global = self.motion_text_proj(global_feat)
            motion_query = self.motion_all_proj(cls_token)
            # if self.ifdown_sample:
            #     motion_mask = motion_mask[:, ::4]
            # 跨模态注意力
            for layer in self.cross_attn_layers:
                text_query = layer(
                    tgt=text_query,
                    memory=motion_query,
                    tgt_mask=None,
                    memory_mask=None,
                    memory_key_padding_mask=~motion_mask,
                    tgt_key_padding_mask=~attention_mask,
                )
                
            # MLM预测
            logits = self.mlm_head(text_query)
            
            # 标签平滑的MLM损失
            # loss_fct = LabelSmoothingCrossEntropy(smoothing=0.05)
            # active_loss = (labels != -100).view(-1)
            # active_logits = logits.view(-1, self.vocab_size)[active_loss]
            # active_labels = labels.view(-1)[active_loss]
            # mlm_loss = loss_fct(active_logits, active_labels.long())
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            active_loss = (labels != -100).view(-1)
            active_logits = logits.view(-1, self.vocab_size)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            mlm_loss = loss_fct(active_logits, active_labels.long())
            # 对比损失
            text_feature_pooler = self.text_motion_proj(text_feature_pooler)
            contrastive_loss = self.contrastive_loss(motion_feature_global, text_feature_pooler, text_id)
        else:
            contrastive_loss = torch.tensor(0.0).to(motion.device)
            mlm_loss = torch.tensor(0.0).to(motion.device)
        
        return cls_token, [contrastive_loss, mlm_loss], [loss_commit, perplexity]

def create_causal_mask(seq_len, device):
    """创建严格因果掩码（禁止关注未来帧）"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
    return mask.to(device)  # [seq_len, seq_len]

def create_cross_causal_mask(tgt_len, src_len, device):
    """
    生成跨注意力因果掩码
    :param tgt_len: 目标序列长度（降采样后）
    :param src_len: 源序列长度（原始输入）
    :return: 掩码矩阵 [tgt_len, src_len]
    """
    # 计算降采样步长（假设能整除）
    stride = src_len // tgt_len  
    
    # 向量化生成掩码
    rows = torch.arange(tgt_len, device=device)[:, None]
    cols = torch.arange(src_len, device=device)[None, :]
    
    # 每个目标位置i最多能看到源的前(i+1)*stride个位置
    mask = cols >= (rows + 1) * stride  # [tgt_len, src_len]
    return mask

class CausalDownsample(nn.Module):
    def __init__(self, dim, down_ratio=2):
        super().__init__()
        self.down_ratio = down_ratio
        self.query = nn.Parameter(torch.randn(196 // (2**down_ratio), dim))
        
        self.causal_conv = nn.ModuleList([
            TemporalDownsampler(dim, causal=True) for _ in range(down_ratio)
        ])
        # 因果多头注意力
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )

    def forward(self, x, padding_mask=None):
        # 步骤1：因果卷积初步降采样
        x_conv = x.clone()
        for layer in self.causal_conv:
            x_conv = layer(x_conv)  # [bs, seq//4, dim]
        
        # 步骤2：因果交叉注意力精调
        causal_mask = create_cross_causal_mask(x_conv.size(1), x.size(1), x.device)
        
        attn_out, _ = self.attn(
            query=x_conv, 
            key=x, 
            value=x,
            attn_mask=causal_mask,        # 时序因果约束
            key_padding_mask=padding_mask # 输入Padding处理
        )
        return attn_out

class ContrastiveLossWithSTSV2(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, motion_feat, text_feat, text_id):
        """
        motion_feat: [B, D] 动作特征
        text_feat: [B, D] 文本特征
        texts: List[str] 原始文本
        """
        pos_mask = torch.zeros(len(text_id), len(text_id), device=motion_feat.device)
        for i in range(len(text_id)):
            for j in range(len(text_id)):
                if text_id[i] == text_id[j]:
                    pos_mask[i, j] = 1
        # 特征归一化
        motion_feat = F.normalize(motion_feat, dim=-1)
        text_feat = F.normalize(text_feat, dim=-1)
        
        # 计算logits
        logits_per_motion = torch.mm(motion_feat, text_feat.T) / self.temperature  # [B, B]
        logits_per_text = logits_per_motion.T
        
        # 多正样本对比损失
        exp_logits = torch.exp(logits_per_motion)
        numerator = torch.sum(exp_logits * pos_mask, dim=1)  # 分子：正样本相似度
        denominator = torch.sum(exp_logits, dim=1)          # 分母：所有样本
        
        # 避免除零
        valid_pos = (pos_mask.sum(dim=1) > 0)
        loss_motion = -torch.log(numerator[valid_pos]/denominator[valid_pos]).mean()
        
        # 对称文本到动作损失
        exp_logits_text = torch.exp(logits_per_text)
        numerator_text = torch.sum(exp_logits_text * pos_mask, dim=1)
        denominator_text = torch.sum(exp_logits_text, dim=1)
        loss_text = -torch.log(numerator_text[valid_pos]/denominator_text[valid_pos]).mean()
        
        return (loss_motion + loss_text) / 2
    
    def compute_disentangle_loss(self, quant_vis, quant_sem, disentanglement_ratio=0.1):
        quant_vis = rearrange(quant_vis, 'b t c -> (b t) c')
        quant_sem = rearrange(quant_sem, 'b t c -> (b t) c')

        quant_vis = F.normalize(quant_vis, p=2, dim=-1)
        quant_sem = F.normalize(quant_sem, p=2, dim=-1)

        dot_product = torch.sum(quant_vis * quant_sem, dim=1)
        loss = torch.mean(dot_product ** 2) * disentanglement_ratio

        return loss

class FeatureJitter(nn.Module):
    """特征加噪"""
    def __init__(self, std=0.1):
        super().__init__()
        self.std = std
        
    def forward(self, x):
        noise = torch.randn_like(x) * self.std
        return x + noise

class ComposeAugmentation(nn.Module):
    """组合时空数据增强"""
    def __init__(self, aug_list):
        super().__init__()
        self.aug_list = aug_list

    def forward(self, x):
        for aug in self.aug_list:
            x = aug(x)
        return x

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        
    def forward(self, x, target):
        confidence = 1. - self.smoothing
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()
    
class TemporalDownsamplerHalf(nn.Module):
    """时间维度1/2降采样模块，使用单层卷积实现"""
    def __init__(self, d_model, causal=False, layer_norm=False):
        super().__init__()
        if layer_norm:
            self.layernorm = nn.LayerNorm(d_model)
        else:
            self.layernorm = nn.Identity()
        if causal:
            self.conv_layers = nn.Sequential(
                RepeatFirstElementPad1d(padding=2),
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=0),
                nn.GELU(),
                # layer
            )
        else:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                # layer
            )
        
    def forward(self, x):
        """
        输入形状: [B, T, C]
        输出形状: [B, T//2, C]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # [B, T//2, C]
        x = self.layernorm(x)
        return x
    
class TemporalDownsampler(nn.Module):
    """时间维度1/4降采样模块，使用单层卷积实现"""
    def __init__(self, d_model, stride_t=2, depth=3, dilation_growth_rate=3, activation='relu', norm=None, causal=True):
        super().__init__()
        filter_t, pad_t = stride_t * 2, stride_t // 2
        causal_pad = filter_t - 1
        self.conv_layers = nn.Sequential(
                nn.ConstantPad1d((causal_pad, 0), 0),  # 左侧填充
                nn.Conv1d(d_model, d_model, filter_t, stride_t, 0),
                Resnet1D(d_model, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal),
            )

    def forward(self, x):
        """
        输入形状: [B, T, C]
        输出形状: [B, T//2, C]
        """
        x = x.permute(0, 2, 1)  # [B, C, T]
        x = self.conv_layers(x)
        x = x.permute(0, 2, 1)  # [B, T//2, C]
        return x    
    
class RepeatFirstElementPad1d(nn.Module):
    """自定义填充层：用第一个元素重复填充左侧"""
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        # x形状: [B, C, T]
        if self.padding == 0:
            return x
        # 取第一个元素并重复填充到左侧
        first_elem = x[:, :, :1]  # [B, C, 1]
        pad = first_elem.repeat(1, 1, self.padding)  # [B, C, padding]
        return torch.cat([pad, x], dim=2)  # [B, C, padding + T]  

class RepeatZeroElementPad1d(nn.Module):
    """自定义填充层：用0填充左侧"""
    def __init__(self, padding):
        super().__init__()
        self.padding = padding

    def forward(self, x):
        # x形状: [B, C, T]
        if self.padding == 0:
            return x
        # 取0并重复填充到左侧
        pad = torch.zeros(x.size(0), x.size(1), self.padding).to(x.device)  # [B, C, padding]
        return torch.cat([pad, x], dim=2)  # [B, C, padding + T]  