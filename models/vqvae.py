import torch
import torch.nn as nn
from models.encdec import LGVQ, Dualsem_encoderv3, Encoder, Decoder, Encoder_Transformer, Decoder_wo_upsample
from models.quantize_cnn import QuantizeEMAReset, Quantizer, QuantizeEMA, QuantizeReset


class VQVAE_251(nn.Module):
    def __init__(self,
                 args,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 enc='cnn',
                 lgvq=0,
                 causal=False,
                 dec_causal=False):
        
        super().__init__()
        self.code_dim = code_dim
        self.num_code = nb_code
        self.quant = args.quantizer
        self.lgvq = lgvq
        if enc == 'cnn':
            print("enc == cnn, causal = ", causal)
            if 'down_vqvae' in args and args.down_vqvae == 1:
                self.encoder = Encoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal)
                self.decoder = Decoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=dec_causal)
            else:
                self.encoder = Encoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, 1, width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal)
                self.decoder = Decoder_wo_upsample(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        else:
            if 'down_vqvae' not in args:
                args.down_vqvae = 0
            print("enc == transformer, causal = ", causal, "down_vqvae = ", args.down_vqvae)
            self.encoder = Encoder_Transformer(dim = 251 if args.dataname == 'kit' else 263, d_model=output_emb_width, num_layers = 2, down_sample=args.down_vqvae if 'down_vqvae' in args else False)
            if 'down_vqvae' in args and args.down_vqvae == 1:
                self.decoder = Decoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=dec_causal)
            else:
                self.decoder = Decoder_wo_upsample(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        if args.quantizer == "ema_reset":
            self.quantizer = QuantizeEMAReset(nb_code, code_dim, args)
        elif args.quantizer == "orig":
            self.quantizer = Quantizer(nb_code, code_dim, 1.0)
        elif args.quantizer == "ema":
            self.quantizer = QuantizeEMA(nb_code, code_dim, args)
        elif args.quantizer == "reset":
            self.quantizer = QuantizeReset(nb_code, code_dim, args)
        self.args = args
        if self.lgvq == 1:
            self.lgvq_encoder = Dualsem_encoderv3(args, d_model=output_emb_width, num_layers=2, down_sample=args.down_sample if 'down_sample' in args else 0)
        elif self.lgvq == 2:
            self.lgvq_encoder = LGVQ(args, d_model=output_emb_width, num_layers=2, down_sample=args.down_sample if 'down_sample' in args else 0, layer_norm=args.layer_norm if 'layer_norm' in args else False)
            


    def preprocess(self, x):
        return x.permute(0,2,1).float()

    def postprocess(self, x):
        return x.permute(0,2,1)


    def encode(self, x):
        N, T, _ = x.shape           # torch.Size([1, 104, 263])
        x_in = self.preprocess(x)   # torch.Size([1, 263, 104])
        x_encoder = self.encoder(x_in) 
        x_encoder = self.postprocess(x_encoder)  # torch.Size([1, 512, 26])
        
        x_encoder_ = x_encoder.contiguous().view(-1, x_encoder.shape[-1])  # (NT, C)
        code_idx = self.quantizer.quantize(x_encoder_)   # torch.Size([26, 512])
        code_idx = code_idx.view(N, -1)
        if self.lgvq == 1:
            sem_idx = self.lgvq_encoder.encode(x_encoder)
            return code_idx, sem_idx
        else:
            return code_idx

    def text_motion_topk(self, motion, text, motion_mask=None, topk=5, text_mask=None):
        x_in = self.preprocess(motion)
        # Encode
        x_encoder = self.encoder(x_in, motion_mask)
        motion = x_encoder.permute(0,2,1)
        # breakpoint()
        # motion_mask = motion_mask[:, ::4].clone()
        return self.lgvq_encoder.text_motion_topk(motion, text, motion_mask=motion_mask, topk=topk, text_mask=text_mask)


    def forward(self, x, motion_mask = None, text_mask = None, text_id = None):
        
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in, motion_mask)
        if self.lgvq == 1:
            if self.args.down_vqvae == 1 and motion_mask is not None:
                if self.args.down_t == 2:
                    motion_mask = motion_mask[:, ::4].clone()
                else:
                    motion_mask = motion_mask[:, ::2].clone()
            cls_token, loss_lgvq, sem_quantized = self.lgvq_encoder(x_encoder.permute(0,2,1), text_mask=text_mask, motion_mask=motion_mask, text_id=text_id)
            contrastive_loss, mlm_loss = loss_lgvq
            loss_sem, perplexity_sem = sem_quantized
        elif self.lgvq == 0:
            contrastive_loss, mlm_loss, loss_sem, perplexity_sem = torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0)
        ## quantization
        x_quantized, loss, perplexity  = self.quantizer(x_encoder)
        if self.lgvq == 2:
            if self.args.down_vqvae == 1 and motion_mask is not None:
                if self.args.down_t == 2:
                    motion_mask = motion_mask[:, ::4].clone()
                else:
                    motion_mask = motion_mask[:, ::2].clone()
            cls_token, loss_lgvq, sem_quantized = self.lgvq_encoder(x_quantized.permute(0,2,1), text_mask=text_mask, motion_mask=motion_mask, text_id=text_id, layer_norm=self.args.layer_norm)
            contrastive_loss, mlm_loss = loss_lgvq
            loss_sem, perplexity_sem = sem_quantized
        ## decoder
        x_decoder = self.decoder(x_quantized)
        x_out = self.postprocess(x_decoder)
        return x_out, loss + loss_sem, perplexity + perplexity_sem, [contrastive_loss, mlm_loss]


    def forward_decoder(self, x):
        x_d = self.quantizer.dequantize(x)
        x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        
        # decoder
        x_decoder = self.decoder(x_d)
        x_out = self.postprocess(x_decoder)
        return x_out
    
    # def text_motion_topk(self, motion, text, text_mask=None, motion_mask=None, topk=5):
    #     x_in = self.preprocess(motion)
    #     x_encoder = self.encoder(x_in, motion_mask)
    #     if self.lgvq == 1:
    #         return self.lgvq_encoder.text_motion_topk(self.preprocess(x_encoder), text, text_mask, motion_mask, topk)
    #     else:
    #         return [], []



class HumanVQVAE(nn.Module):
    def __init__(self,
                 args,
                 nb_code=512,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                 enc='cnn',
                 lgvq=0,
                 causal=False,
                 dec_causal=False):
        
        super().__init__()
        
        self.nb_joints = 21 if args.dataname == 'kit' else 22
        self.vqvae = VQVAE_251(args, nb_code, code_dim, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm, enc=enc, lgvq=lgvq, causal=causal, dec_causal=dec_causal)

    def encode(self, x):
        b, t, c = x.size()
        quants = self.vqvae.encode(x) # (N, T)
        return quants

    def forward(self, x, motion_mask = None, text_mask = None, text_id = None):

        x_out, loss, perplexity, loss_lgvq = self.vqvae(x, motion_mask, text_mask, text_id)
        
        return x_out, loss, perplexity, loss_lgvq

    def forward_decoder(self, x):
        x_out = self.vqvae.forward_decoder(x)
        return x_out
    
    def text_motion_topk(self, motion, text, text_mask=None, motion_mask=None, topk=5):
        return self.vqvae.text_motion_topk(motion, text, text_mask, motion_mask, topk)

import torch.nn as nn
from models.encdec import Encoder, Decoder
from models.vq.residual_vq import ResidualVQ
    
class RVQVAE(nn.Module):
    def __init__(self,
                 args,
                 input_width=263,
                 nb_code=1024,
                 code_dim=512,
                 output_emb_width=512,
                 down_t=3,
                 stride_t=2,
                 width=512,
                 depth=3,
                 dilation_growth_rate=3,
                 activation='relu',
                 norm=None,
                #  lgvq=0,
                 causal=True):

        super().__init__()
        assert output_emb_width == code_dim
        self.args = args
        self.code_dim = code_dim
        self.num_code = nb_code
        if args.down_vqvae == 1:
            self.encoder = Encoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm, causal=causal)
            self.decoder = Decoder(input_width, output_emb_width, down_t, stride_t, width, depth,
                                dilation_growth_rate, activation=activation, norm=norm)
        else:
            self.encoder = Encoder(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, 1, width, depth, dilation_growth_rate, activation=activation, norm=norm, causal=causal)
            self.decoder = Decoder_wo_upsample(251 if args.dataname == 'kit' else 263, output_emb_width, down_t, stride_t, width, depth, dilation_growth_rate, activation=activation, norm=norm)
        rvqvae_config = {
            'num_quantizers': args.num_quantizers,
            'shared_codebook': args.shared_codebook,
            'quantize_dropout_prob': args.quantize_dropout_prob,
            'quantize_dropout_cutoff_index': 0,
            'nb_code': nb_code,
            'code_dim':code_dim, 
            'args': args,
        }
        self.quantizer = ResidualVQ(**rvqvae_config)
        
        if self.args.lgvq == 1:
            self.lgvq_encoder = Dualsem_encoderv3(args, d_model=output_emb_width, num_layers=2, down_sample=args.down_sample if 'down_sample' in args else 0)

    def preprocess(self, x):
        # (bs, T, Jx3) -> (bs, Jx3, T)
        x = x.permute(0, 2, 1).float()
        return x

    def postprocess(self, x):
        # (bs, Jx3, T) ->  (bs, T, Jx3)
        x = x.permute(0, 2, 1)
        return x

    def encode(self, x):
        N, T, _ = x.shape
        x_in = self.preprocess(x)
        x_encoder = self.encoder(x_in)
        code_idx, all_codes = self.quantizer.quantize(x_encoder, return_latent=True)
        if self.args.lgvq == 1:
            x_encoder = self.lgvq_encoder.encode(x_encoder)
            return code_idx, all_codes, x_encoder
        # print(x_encoder.shape)
        
        # print(code_idx.shape)
        # code_idx = code_idx.view(N, -1)
        # (N, T, Q)
        # print()
        return code_idx, all_codes

    def forward(self, x, motion_mask = None, text_mask = None, text_id = None):
        x_in = self.preprocess(x)
        # Encode
        x_encoder = self.encoder(x_in)
        x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5)
        if self.args.lgvq == 1:
            # x_encoder = self.lgvq_encoder(x_encoder)
            cls_token, loss_lgvq, sem_quantized = self.lgvq_encoder(x_encoder.permute(0,2,1), text_mask=text_mask, motion_mask=motion_mask, text_id=text_id)
            loss_sem, perplexity_sem = sem_quantized
            commit_loss = commit_loss + loss_sem
            perplexity = perplexity + perplexity_sem
            contrastive_loss, mlm_loss = loss_lgvq
        else:
            contrastive_loss, mlm_loss = torch.tensor(0), torch.tensor(0)
        ## quantization
        # x_quantized, code_idx, commit_loss, perplexity = self.quantizer(x_encoder, sample_codebook_temp=0.5,
        #                                                                 force_dropout_index=0) #TODO hardcode
        # print(code_idx[0, :, 1])
        ## decoder
        x_out = self.decoder(x_quantized)
        x_out = self.postprocess(x_out)
        return x_out, commit_loss, perplexity, [contrastive_loss, mlm_loss]

    def forward_decoder(self, x):
        x_d = self.quantizer.get_codes_from_indices(x)
        # x_d = x_d.view(1, -1, self.code_dim).permute(0, 2, 1).contiguous()
        x = x_d.sum(dim=0).permute(0, 2, 1)

        # decoder
        x_out = self.decoder(x)
        # x_out = self.postprocess(x_decoder)
        return x_out
