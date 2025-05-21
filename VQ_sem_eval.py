import os
import json

import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import models.vqvae as vqvae
import options.option_vq as option_vq
import utils.utils_model as utils_model
from dataset import dataset_TM_eval
import utils.eval_trans as eval_trans
from options.get_eval_option import get_opt
from models.evaluator_wrapper import EvaluatorModelWrapper
import warnings
warnings.filterwarnings('ignore')
import numpy as np
##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)
path = os.path.dirname(args.resume_pth)
json_file = os.path.join(path, 'train_config.json')
model_path = os.path.join(path, 'net_best_fid.pth')
with open(json_file, 'r') as f:
    train_args_dict = json.load(f)  # dict
args_vq = eval_trans.EasyDict(train_args_dict)

args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
os.makedirs(args.out_dir, exist_ok = True)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.out_dir)
writer = SummaryWriter(args.out_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))


from utils.word_vectorizer import WordVectorizer
w_vectorizer = WordVectorizer('./glove', 'our_vab')


dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
eval_wrapper = EvaluatorModelWrapper(wrapper_opt)


##### ---- Dataloader ---- #####
args.nb_joints = 21 if args.dataname == 'kit' else 22

val_loader = dataset_TM_eval.DATALoader(args.dataname, True, 32, w_vectorizer, unit_length=2**args.down_t)


##### ---- Network ---- #####
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
                       args_vq.vq_norm,
                       enc=args_vq.enc,
                       lgvq=args_vq.lgvq,
                       causal=args_vq.causal if 'causal' in args_vq else 0,
                       dec_causal=args_vq.dec_causal if 'dec_causal' in args_vq else 0)


logger.info('loading checkpoint from {}'.format(model_path))
ckpt = torch.load(model_path, map_location='cpu')
net.load_state_dict(ckpt['net'], strict=True)

net.eval()
net.cuda()

fid = []
div = []
top1 = []
top2 = []
top3 = []
matching = []
best_mpjpe_list = []
repeat_time = 5
for i in range(repeat_time):
    best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, best_mpjpe = eval_trans.evaluation_vqvae(args.out_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, draw=False, save=False, savenpy=(i==0), best_mpjpe=1e9)
    fid.append(best_fid)
    div.append(best_div)
    top1.append(best_top1)
    top2.append(best_top2)
    top3.append(best_top3)
    matching.append(best_matching)
    # Handle both tensor and int/float cases
    if isinstance(best_mpjpe, torch.Tensor):
        best_mpjpe_list.append(best_mpjpe.cpu().numpy())
    else:
        best_mpjpe_list.append(best_mpjpe)
print('final result:')
print('fid: ', sum(fid)/repeat_time)
print('div: ', sum(div)/repeat_time)
print('top1: ', sum(top1)/repeat_time)
print('top2: ', sum(top2)/repeat_time)
print('top3: ', sum(top3)/repeat_time)
print('matching: ', sum(matching)/repeat_time)
print('mpjpe: ', sum(best_mpjpe_list)/repeat_time)
fid = np.array(fid)
div = np.array(div)
top1 = np.array(top1)
top2 = np.array(top2)
top3 = np.array(top3)
matching = np.array(matching)
mpjpe = np.array(best_mpjpe_list)
msg_final = f"FID. {np.mean(fid):.3f}, conf. {np.std(fid)*1.96/np.sqrt(repeat_time):.3f}, Diversity. {np.mean(div):.3f}, conf. {np.std(div)*1.96/np.sqrt(repeat_time):.3f}, TOP1. {np.mean(top1):.3f}, conf. {np.std(top1)*1.96/np.sqrt(repeat_time):.3f}, TOP2. {np.mean(top2):.3f}, conf. {np.std(top2)*1.96/np.sqrt(repeat_time):.3f}, TOP3. {np.mean(top3):.3f}, conf. {np.std(top3)*1.96/np.sqrt(repeat_time):.3f}, Matching. {np.mean(matching):.3f}, conf. {np.std(matching)*1.96/np.sqrt(repeat_time):.3f}, MPJPE. {np.mean(mpjpe):.3f}, conf. {np.std(mpjpe)*1.96/np.sqrt(repeat_time):.3f}"
logger.info(msg_final)