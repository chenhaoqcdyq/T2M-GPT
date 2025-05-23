import os
import json
import re
import time
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
path = "output/00170-t2m-VQVAE_cnn_down_lgvq/VQVAE-VQVAE_cnn_down_lgvq-t2m"
# path = "output/00211-t2m-VQVAE_cnn_wodown_lgvq/VQVAE-VQVAE_cnn_wodown_lgvq-t2m"
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
timestamp = time.strftime("%Y%m%d-%H%M%S")
logs_dir = os.path.join(args.run_dir, f'logs_{timestamp}')
os.makedirs(logs_dir, exist_ok=True)
logger = utils_model.get_logger(args.run_dir,logs_dir)
# Create timestamped subdirectories in run_dir

writer = SummaryWriter(logs_dir)
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
        print("dataset_VQ_all_text")
    else:
        import dataset.dataset_VQ_all as dataset_VQ
        print("dataset_VQ_all")
else:
    import dataset.dataset_VQ as dataset_VQ
    print("dataset_VQ")
##### ---- Dataloader ---- #####
test_loader = dataset_VQ.DATALoader(args.dataname,
                                        32,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        test=True)

test_loader_iter = dataset_VQ.cycle(test_loader)

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

# Initialize lists to store results across multiple runs
R1_all_runs = []
R2_all_runs = []
repeat_time = 5

print(f"Starting evaluation with {repeat_time} repetitions...")

for run_idx in range(repeat_time):
    print(f"Run {run_idx + 1}/{repeat_time}")
    
    # Reset lists for current run
    R1_current_run = []
    R2_current_run = []
    
    # Evaluate on the entire test dataset for current run
    for i in tqdm(range(len(test_loader)), desc=f"Run {run_idx + 1}"):
        gt_motion = next(test_loader_iter)
        if isinstance(gt_motion, tuple) or isinstance(gt_motion, list):
            if len(gt_motion) == 2:
                gt_motion, gt_motion_mask = gt_motion
                text_mask, name, text = None, None, None
            elif len(gt_motion) == 3:
                gt_motion, gt_motion_mask, text_mask = gt_motion
                name, text = None, None
            elif len(gt_motion) == 4:
                gt_motion, gt_motion_mask, text_mask, name = gt_motion
                text = None
            elif len(gt_motion) == 5:
                gt_motion, gt_motion_mask, text_mask, name, text = gt_motion
        else:
            gt_motion_mask, text_mask, name, text = None, None, None, None
            
        gt_motion = gt_motion.cuda().float() # (bs, 64, dim)
        if gt_motion_mask is not None:
            gt_motion_mask = gt_motion_mask.cuda().long() # (bs, 64)
        
        with torch.no_grad():
            result = net.text_motion_topk(gt_motion, motion_mask=gt_motion_mask, topk=5, text_mask=text_mask, text=text)
        
        global_R, pred_R = result
        R1_current_run.append(global_R)
        R2_current_run.append(pred_R)
    
    # Calculate mean for current run
    R1_run_mean = np.mean(np.array(R1_current_run), axis=0)
    R2_run_mean = np.mean(np.array(R2_current_run), axis=0)
    
    R1_all_runs.append(R1_run_mean)
    R2_all_runs.append(R2_run_mean)
    
    print(f"Run {run_idx + 1} - R1 mean: {R1_run_mean}")
    print(f"Run {run_idx + 1} - R2 mean: {R2_run_mean}")

# Calculate final statistics across all runs
R1_all_runs = np.array(R1_all_runs)
R2_all_runs = np.array(R2_all_runs)

# Calculate mean and confidence intervals
R1_final_mean = np.mean(R1_all_runs, axis=0)
R2_final_mean = np.mean(R2_all_runs, axis=0)

R1_std = np.std(R1_all_runs, axis=0)
R2_std = np.std(R2_all_runs, axis=0)

R1_conf = R1_std * 1.96 / np.sqrt(repeat_time)
R2_conf = R2_std * 1.96 / np.sqrt(repeat_time)

print("\n" + "="*50)
print("FINAL RESULTS:")
print("="*50)
print(f"R1 final mean: {R1_final_mean}")
print(f"R1 confidence interval: ±{R1_conf}")
print(f"R2 final mean: {R2_final_mean}")
print(f"R2 confidence interval: ±{R2_conf}")

# Log detailed results
logger.info("="*50)
logger.info("FINAL EVALUATION RESULTS")
logger.info("="*50)
logger.info(f"Number of repetitions: {repeat_time}")
logger.info(f"R1 final mean: {R1_final_mean}")
logger.info(f"R1 std: {R1_std}")
logger.info(f"R1 confidence interval (95%): ±{R1_conf}")
logger.info(f"R2 final mean: {R2_final_mean}")
logger.info(f"R2 std: {R2_std}")
logger.info(f"R2 confidence interval (95%): ±{R2_conf}")

# Format final message similar to VQ_sem_eval.py
if len(R1_final_mean) >= 3 and len(R2_final_mean) >= 3:
    msg_final = (f"R1 Results - "
                f"Top1: {R1_final_mean[0]:.3f}, conf: {R1_conf[0]:.3f}, "
                f"Top2: {R1_final_mean[1]:.3f}, conf: {R1_conf[1]:.3f}, "
                f"Top3: {R1_final_mean[2]:.3f}, conf: {R1_conf[2]:.3f}, "
                f"R2 Results - "
                f"Top1: {R2_final_mean[0]:.3f}, conf: {R2_conf[0]:.3f}, "
                f"Top2: {R2_final_mean[1]:.3f}, conf: {R2_conf[1]:.3f}, "
                f"Top3: {R2_final_mean[2]:.3f}, conf: {R2_conf[2]:.3f}, "
                )
else:
    msg_final = (f"R1 Results: {R1_final_mean}, conf: {R1_conf} | "
                f"R2 Results: {R2_final_mean}, conf: {R2_conf}")

logger.info(msg_final)
print("\nFormatted Results:")
print(msg_final)