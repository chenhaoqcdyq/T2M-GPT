import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.distributions import Categorical
import models.pos_encoding as pos_encoding

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
                semantic_flag=False):
        super().__init__()
        self.trans_base = CrossCondTransBase(num_vq, embed_dim, clip_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.trans_head = CrossCondTransHead(num_vq, embed_dim, block_size, num_layers, n_head, drop_out_rate, fc_rate)
        self.block_size = block_size
        self.num_vq = num_vq
        # self.reconstruction_flag = 0
        self.semantic_flag = semantic_flag
        # Define token ID constants
        # Assuming self.num_vq is the STOP_TOKEN_ID for the combined (semantic+separator+reconstruction) vocabulary.
        # Example: self.num_vq = 1025 means tokens 0-1024 are valid, 1025 is STOP.
        # If semantic and reconstruction codebooks are each 512:
        # Semantic tokens: 0-511
        # Separator token: 512
        # Reconstruction tokens: 513-1024
        # This implies _assumed_codebook_part_size = 512.
        # We derive _assumed_codebook_part_size from self.num_vq to be robust.
        # (self.num_vq - 1 (separator) - 1 (stop_token_id_is_size_not_index)) / 2
        # If self.num_vq = 1025 (stop token ID), then max active token index is 1024.
        # Number of active tokens = self.num_vq.
        # If vocabulary is [sem (N), sep (1), recon (N)], total active tokens = 2N+1. Stop token is 2N+1.
        # So, self.num_vq = 2N+1. N = (self.num_vq - 1) / 2.
        _assumed_codebook_part_size = (self.num_vq - 1) // 2 # e.g. (1025-1)//2 = 512

        self.SEMANTIC_TOKEN_END_IDX = _assumed_codebook_part_size - 1
        self.SEPARATOR_TOKEN_IDX = _assumed_codebook_part_size
        self.RECONSTRUCTION_TOKEN_START_IDX = _assumed_codebook_part_size + 1
        # RECONSTRUCTION_TOKEN_END_IDX is self.num_vq - 1 (the max token ID before stop)
        # STOP_TOKEN_ID is self.num_vq

    def get_block_size(self):
        return self.block_size

    def forward(self, idxs, clip_feature):
        feat = self.trans_base(idxs, clip_feature)
        logits = self.trans_head(feat)
        return logits

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
        if not self.semantic_flag:
            # Use the original sampling logic
            return self.sample_original_backup(clip_feature, if_categorial)
        else:
            # Semantic-aware sampling logic
            xs = None
            # semantic_flag is True, start with semantic phase
            current_sampling_phase = "semantic" 
            sampled_at_least_one_reconstruction_token = False
            
            # Loop to generate up to block_size - 1 tokens
            # (condition clip_feature takes one spot in the transformer's view, 
            # effectively allowing block_size-1 generated tokens to form a sequence of self.block_size with condition)
            for k_loop_iter in range(self.block_size -1): 
                current_input_for_transformer = xs if xs is not None else []
                
                logits = self.forward(current_input_for_transformer, clip_feature)
                logits = logits[:, -1, :] 
                probs = F.softmax(logits, dim=-1)

                # Mask probabilities based on the current sampling phase
                # self.num_vq is the ID of the stop token. Max actual data token ID is self.num_vq - 1.
                temp_probs = probs.clone()

                if current_sampling_phase == "semantic":
                    # Allow [0, SEMANTIC_TOKEN_END_IDX] or SEPARATOR_TOKEN_IDX
                    # Block [RECONSTRUCTION_TOKEN_START_IDX, self.num_vq-1 (max_token_id)]
                    temp_probs[:, self.RECONSTRUCTION_TOKEN_START_IDX : self.num_vq] = 0 
                    # Also block explicit stop if in semantic phase before separator
                    temp_probs[:, self.num_vq] = 0 
                     # If all semantic token probabilities are extremely low, force SEPARATOR_TOKEN_IDX
                    if torch.sum(temp_probs[:, :self.SEPARATOR_TOKEN_IDX]) < 1e-9 :
                        temp_probs[:, :self.SEPARATOR_TOKEN_IDX+1] = 0 # Zero out semantic and separator initially
                        temp_probs[:, self.SEPARATOR_TOKEN_IDX] = 1.0 # Force separator

                elif current_sampling_phase == "reconstruction":
                    # Allow [RECONSTRUCTION_TOKEN_START_IDX, self.num_vq-1] or STOP_TOKEN (self.num_vq)
                    # Block all semantic tokens and the separator token
                    temp_probs[:, :self.RECONSTRUCTION_TOKEN_START_IDX] = 0
                
                # Renormalize probabilities
                batch_sums = temp_probs.sum(dim=-1, keepdim=True)
                # Handle cases where all valid probabilities become zero after masking
                for i_batch in range(temp_probs.shape[0]):
                    if batch_sums[i_batch] < 1e-9: # If sum is too small
                        if current_sampling_phase == "semantic":
                            # Fallback: force SEPARATOR_TOKEN_IDX
                            temp_probs[i_batch, :] = 0
                            temp_probs[i_batch, self.SEPARATOR_TOKEN_IDX] = 1.0
                        else: # reconstruction phase
                            # Fallback: force STOP_TOKEN_ID
                            temp_probs[i_batch, :] = 0
                            temp_probs[i_batch, self.num_vq] = 1.0 
                # Re-calculate sums after potential fallback
                batch_sums = temp_probs.sum(dim=-1, keepdim=True)
                # Avoid division by zero if somehow still all zeros (should be caught by fallback)
                probs_masked = temp_probs / (batch_sums + 1e-9)


                # Sample token from masked probabilities
                idx_sampled_token_val = None
                if if_categorial:
                    dist = Categorical(probs_masked)
                    idx_val = dist.sample()
                    if idx_val == self.num_vq: # Stop token
                        break
                    idx_sampled_token_val = idx_val.unsqueeze(-1)
                else:
                    _, idx_val_topk = torch.topk(probs_masked, k=1, dim=-1)
                    if idx_val_topk[0,0] == self.num_vq: # Stop token
                        break
                    idx_sampled_token_val = idx_val_topk

                # Append to sequence
                if xs is None:
                    xs = idx_sampled_token_val
                else:
                    xs = torch.cat((xs, idx_sampled_token_val), dim=1)

                # Update phase and flags
                sampled_token_id = idx_sampled_token_val[0,0].item()
                if current_sampling_phase == "semantic" and sampled_token_id == self.SEPARATOR_TOKEN_IDX:
                    current_sampling_phase = "reconstruction"
                elif current_sampling_phase == "reconstruction":
                    if self.RECONSTRUCTION_TOKEN_START_IDX <= sampled_token_id < self.num_vq:
                        sampled_at_least_one_reconstruction_token = True
            
            # After loop, ensure reconstruction token if needed
            if current_sampling_phase == "reconstruction" and not sampled_at_least_one_reconstruction_token:
                # If sequence is empty or only separator, add a reconstruction token
                if xs is None or xs.shape[1] == 0 :
                    xs = torch.tensor([[self.RECONSTRUCTION_TOKEN_START_IDX]], dtype=torch.long, device=clip_feature.device)
                elif xs[0,-1].item() == self.SEPARATOR_TOKEN_IDX: # Ends with separator
                    # Append a reconstruction token if space allows (block_size-1 max generated tokens)
                    if xs.shape[1] < (self.block_size -1):
                        xs = torch.cat((xs, torch.tensor([[self.RECONSTRUCTION_TOKEN_START_IDX]], dtype=torch.long, device=clip_feature.device)), dim=1)
                    else: # No space, replace separator
                        xs[0,-1] = self.RECONSTRUCTION_TOKEN_START_IDX
                elif xs[0,-1].item() != self.num_vq : # Last token is not STOP and not SEPARATOR
                    xs[0,-1] = self.RECONSTRUCTION_TOKEN_START_IDX # Replace last token

            if xs is None:
                return torch.empty((clip_feature.shape[0], 0), dtype=torch.long, device=clip_feature.device)
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

    def forward(self, x):
        B, T, C = x.size() 

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:T,:T] == 0, float('-inf'))
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

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
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
        self.tok_emb = nn.Embedding(num_vq + 2, embed_dim)
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
    
    def forward(self, idx, clip_feature):
        if len(idx) == 0:
            token_embeddings = self.cond_emb(clip_feature).unsqueeze(1)
        else:
            b, t = idx.size()
            assert t <= self.block_size, "Cannot forward, model block size is exhausted."
            # forward the Trans model
            token_embeddings = self.tok_emb(idx)
            token_embeddings = torch.cat([self.cond_emb(clip_feature).unsqueeze(1), token_embeddings], dim=1)
            
        x = self.pos_embed(token_embeddings)
        x = self.blocks(x)

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

    


        

