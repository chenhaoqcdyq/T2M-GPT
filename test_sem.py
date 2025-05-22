import os
import json
import re

import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import models.vqvae as vqvae
import utils.losses as losses 
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import  dataset_TM_eval

import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
from utils.word_vectorizer import WordVectorizer
torch.autograd.set_detect_anomaly(True)
def freeze_encdec(net):
    for name, param in net.named_parameters():
        if ('lgvq_encoder' in name) and "bert_model" not in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    for name, module in net.named_modules():
        if ('lgvq_encoder' in name) and "bert_model" not in name:
            module.train()
        else:
            module.eval()
    return net

def update_lr_warm_up(optimizer, nb_iter, warm_up_iter, lr):

    current_lr = lr * (nb_iter + 1) / (warm_up_iter + 1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = current_lr

    return optimizer, current_lr

args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
##### ---- Exp dirs ---- #####
path = "output/00152-t2m-VQVAE_lgvq_cnn_down/VQVAE-VQVAE_lgvq_cnn_down-t2m"
json_file = os.path.join(path, 'train_config.json')
checkpoint_path = os.path.join(path, 'net_last.pth')
with open(json_file, 'r') as f:
    train_args_dict = json.load(f)  # dict
args = eval_trans.EasyDict(train_args_dict) 
# args = option_vq.get_args_parser()

# desc = args.dataname  # dataset
# # desc += f'-{args.exp_name}'
# outdir = args.out_dir
# if os.path.isdir(outdir):
#     prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
# prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
# prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
# cur_run_id = max(prev_run_ids, default=-1) + 1
# args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{args.dataname}-{args.exp_name}', f'VQVAE-{args.exp_name}-{desc}')
# assert not os.path.exists(args.run_dir)
# print('Creating output directory...')
# os.makedirs(args.run_dir)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.run_dir)
writer = SummaryWriter(args.run_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))



w_vectorizer = WordVectorizer('./glove', 'our_vab')

if args.dataname == 'kit' : 
    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt'  
    args.nb_joints = 21
    
else :
    dataset_opt_path = 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'
    args.nb_joints = 22

logger.info(f'Training on {args.dataname}, motions are with {args.nb_joints} joints')

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)
if args.all_motion:
    if args.lgvq == 1:
        import dataset.dataset_VQ_all_text as dataset_VQ
    else:
        import dataset.dataset_VQ_all as dataset_VQ
else:
    import dataset.dataset_VQ as dataset_VQ
##### ---- Dataloader ---- #####
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        32,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

train_loader_iter = dataset_VQ.cycle(train_loader)

# val_loader = dataset_TM_eval.DATALoader(args.dataname, False,
#                                         32,
#                                         w_vectorizer,
#                                         unit_length=2**args.down_t)
print("args = ",args)
net = vqvae.HumanVQVAE(args, ## use args to define different parameters in different quantizers
                       args.nb_code,
                       args.code_dim,
                       args.output_emb_width,
                       args.down_t,
                       args.stride_t,
                       args.width,
                       args.depth,
                       args.dilation_growth_rate,
                       args.vq_act,
                       args.vq_norm,
                       enc=args.enc,
                       lgvq=args.lgvq,
                       causal=args.causal if 'causal' in args else 0,
                       dec_causal=args.dec_causal if 'dec_causal' in args else 0)



logger.info('loading checkpoint from {}'.format(checkpoint_path))
ckpt = torch.load(checkpoint_path, map_location='cpu')
keys = net.load_state_dict(ckpt['net'], strict=False)
print("missing_keys",keys.missing_keys)
print("unexpected_keys",keys.unexpected_keys)
net.eval()
net.cuda()

# if args.freeze_encdec != 0 and args.lgvq != 0:
#     net = freeze_encdec(net)
    
R1 = []
R2 = []
for i in tqdm(range(1)):
    gt_motion = next(train_loader_iter)
    if isinstance(gt_motion, tuple) or isinstance(gt_motion, list):
        if len(gt_motion) == 2:
            gt_motion, gt_motion_mask = gt_motion
            text_mask, name = None, None
        elif len(gt_motion) == 3:
            gt_motion, gt_motion_mask, text_mask = gt_motion
        elif len(gt_motion) == 4:
            gt_motion, gt_motion_mask, text_mask, name = gt_motion
        elif len(gt_motion) == 5:
            gt_motion, gt_motion_mask, text_mask, name, text = gt_motion
    else:
        gt_motion_mask, text_mask, name = None, None, None
    gt_motion = gt_motion.cuda().float() # (bs, 64, dim)
    if gt_motion_mask is not None:
        gt_motion_mask = gt_motion_mask.cuda().long() # (bs, 64)
    
    with torch.no_grad():
        # for i in range(len(gt_parts[0])):
        result = net.text_motion_topk(gt_motion, motion_mask=gt_motion_mask, topk=5, text_mask=text_mask, text=text)
    global_R, pred_R = result
    R1.append(global_R)
    R2.append(pred_R)
    print(result)
R1_mean = np.mean(np.array(R1), axis=0)
R2_mean = np.mean(np.array(R2), axis=0)
print("R1 均值:", R1_mean)
print("R2 均值:", R2_mean)