import os
import re 
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from os.path import join as pjoin
from torch.distributions import Categorical
import json
import clip
from tqdm import tqdm

import options.option_transformer as option_trans
import models.vqvae as vqvae
import utils.utils_model as utils_model
import utils.eval_trans as eval_trans

from dataset import dataset_TM_eval
from dataset import dataset_tokenize
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
args.args_save_dir = os.path.join(args.out_dir, 'train_config.json')
args_dict = vars(args)
with open(args.args_save_dir, 'wt') as f:
    json.dump(args_dict, f, indent=4)

##### ---- Network ---- #####
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


print('loading checkpoint from {}'.format(args.resume_pth))
ckpt = torch.load(args.resume_pth, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)
net.eval()
net.cuda()


##### ---- Dataloader ---- #####
train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=1)

from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')
val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)



if args_vq.lgvq == 1:
    num_vq_trans = args.nb_code * 2 + 1
else:
    num_vq_trans = args.nb_code
trans_encoder = trans.Text2Motion_Transformer(num_vq=num_vq_trans, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate)


if args.resume_trans is not None:
    print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    ckpt = torch.load(args.resume_trans, map_location='cpu')
    trans_encoder.load_state_dict(ckpt['trans'], strict=True)
trans_encoder.train()
trans_encoder.cuda()

##### ---- Optimizer & Scheduler ---- #####
optimizer = utils_model.initial_optim(args.decay_option, args.lr, args.weight_decay, trans_encoder, args.optimizer)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)

##### ---- Optimization goals ---- #####
loss_ce = torch.nn.CrossEntropyLoss()

nb_iter, avg_loss_cls, avg_acc = 0, 0., 0.
right_num = 0
nb_sample_train = 0

# ##### ---- get code ---- #####
# generate_idx = False
for batch in tqdm(train_loader_token):
    pose, name, text = batch
    bs, seq = pose.shape[0], pose.shape[1]

    pose = pose.cuda().float() # bs, nb_joints, joints_dim, seq_len (1,124,263)
    target = net.encode(pose)
    if isinstance(target, tuple):
        motion_idx = target[0]
        sem_idx = target[1]
    else:
        motion_idx = target
        sem_idx = None
    motion_idx = motion_idx.cpu().numpy() # (1, x)

    if sem_idx is not None:
        sem_idx = sem_idx.cpu().numpy() # (1, x)
        np.savez(pjoin(args.vq_dir, name[0] +'.npz'), motion=motion_idx, sem=sem_idx, text=text)
    else:
        np.savez(pjoin(args.vq_dir, name[0] +'.npz'), motion=motion_idx, text=text)