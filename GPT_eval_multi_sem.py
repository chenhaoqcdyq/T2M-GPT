import os
import re 
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import json
import clip

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans
from dataset import dataset_TM_eval
import models.t2m_trans as trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')

##### ---- Exp dirs ---- #####
args = option_trans.get_args_parser()
torch.manual_seed(args.seed)

exp_name = (args.resume_pth).replace('output/','')[:5]
vq_name = f"VQVAE-T2MGPT-SEM-train-{exp_name}"
print("vq_name = ", vq_name)
# args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
desc = args.dataname
outdir = args.out_dir
if os.path.isdir(outdir):
    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
cur_run_id = max(prev_run_ids, default=-1) + 1
args.out_dir = os.path.join(outdir, f'{cur_run_id:05d}-{args.dataname}-{args.exp_name}', f'VQVAE-{args.exp_name}-{desc}')

args.vq_dir= os.path.join("./dataset/KIT-ML" if args.dataname == 'kit' else "./dataset/HumanML3D", f'{vq_name}')
os.makedirs(args.out_dir, exist_ok = True)
os.makedirs(args.vq_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

##### ---- Network ---- #####

## load clip model and datasets
clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
clip_model.eval()
for p in clip_model.parameters():
    p.requires_grad = False

path = os.path.dirname(args.resume_pth)
json_file = os.path.join(path, 'train_config.json')
with open(json_file, 'r') as f:
    train_args_dict = json.load(f)  # dict
args_vq = eval_trans.EasyDict(train_args_dict) 

net = vqvae.HumanVQVAE(args_vq, ## use args to define different parameters in different quantizers
                       args_vq.nb_code,
                       args_vq.code_dim,
                       args_vq.output_emb_width,
                       args_vq.down_t,
                       args_vq.stride_t,
                       args_vq.width,
                       args_vq.depth,
                       args_vq.dilation_growth_rate,
                       args_vq.vq_act,
                    #    args.vq_norm,
                       enc=args_vq.enc,
                       lgvq=args_vq.lgvq,
                       causal=args_vq.causal if 'causal' in args_vq else 0)

trans_path = os.path.dirname(args.resume_trans)
json_file = os.path.join(trans_path, 'train_config.json')
with open(json_file, 'r') as f:
    train_args_dict = json.load(f)  # dict
args_trans = eval_trans.EasyDict(train_args_dict) 
if args_vq.lgvq:
    num_vq_trans = args_vq.nb_code * 2 + 1
else:
    num_vq_trans = args_vq.nb_code
if args_vq.lgvq == 1 and args.sample_way == 0:
    semantic_flag = True
else:
    semantic_flag = False
trans_encoder = trans.Text2Motion_Transformer(num_vq=num_vq_trans, 
                                embed_dim=args_trans.embed_dim_gpt, 
                                clip_dim=args_trans.clip_dim, 
                                block_size=args_trans.block_size, 
                                num_layers=args_trans.num_layers, 
                                n_head=args_trans.n_head_gpt, 
                                drop_out_rate=args_trans.drop_out_rate, 
                                fc_rate=args_trans.ff_rate,
                                semantic_flag=semantic_flag)


print ('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()

if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.eval()
trans_encoder.cuda()


fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
multi = []
repeat_time = 20

        
for i in range(repeat_time):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, writer, logger \
        = eval_trans.evaluation_transformer_test(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, \
            best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, \
                best_multi=0, clip_model=clip_model, eval_wrapper=eval_wrapper, draw=False, savegif=False, save=False, savenpy=(i==0), semantic_flag=args_vq.lgvq)
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
    multi.append(best_multi)

print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)
print('multi: ', sum(multi)/repeat_time)

fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
multi = np.array(multi)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, Multi. {np.mean(multi):.3f}, conf. {np.std(multi)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)