import os
import json
import re

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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
from models.tma.models.architectures.temos.textencoder.distillbert_actor import DistilbertActorAgnosticEncoder
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

##### ---- Exp dirs ---- #####
args = option_vq.get_args_parser()
torch.manual_seed(args.seed)

# args.out_dir = os.path.join(args.out_dir, f'{args.exp_name}')
# os.makedirs(args.out_dir, exist_ok = True)
desc = args.dataname  # dataset
# desc += f'-{args.exp_name}'
outdir = args.out_dir
if os.path.isdir(outdir):
    prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]
prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
cur_run_id = max(prev_run_ids, default=-1) + 1
args.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{args.dataname}-{args.exp_name}', f'VQVAE-{args.exp_name}-{desc}')
assert not os.path.exists(args.run_dir)
print('Creating output directory...')
os.makedirs(args.run_dir)

##### ---- Logger ---- #####
logger = utils_model.get_logger(args.run_dir)
writer = SummaryWriter(args.run_dir)
logger.info(json.dumps(vars(args), indent=4, sort_keys=True))
args.args_save_dir = os.path.join(args.run_dir, 'train_config.json')
args_dict = vars(args)
with open(args.args_save_dir, 'wt') as f:
    json.dump(args_dict, f, indent=4)


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

##### ---- Network ---- #####
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


if args.resume_pth : 
    logger.info('loading checkpoint from {}'.format(args.resume_pth))
    ckpt = torch.load(args.resume_pth, map_location='cpu')
    keys = net.load_state_dict(ckpt['net'], strict=False)
    print("missing_keys",keys.missing_keys)
    print("unexpected_keys",keys.unexpected_keys)
net.train()
net.cuda()

if args.freeze_encdec != 0 and args.lgvq != 0:
    net = freeze_encdec(net)

if args.all_motion:
    if args.lgvq >= 1:
        import dataset.dataset_VQ_all_text as dataset_VQ
    else:
        import dataset.dataset_VQ_all as dataset_VQ
else:
    import dataset.dataset_VQ as dataset_VQ
##### ---- Dataloader ---- #####
train_loader = dataset_VQ.DATALoader(args.dataname,
                                        args.batch_size,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t)

train_loader_iter = dataset_VQ.cycle(train_loader)

val_loader = dataset_TM_eval.DATALoader(args.dataname, False,
                                        32,
                                        w_vectorizer,
                                        unit_length=2**args.down_t)
if args.lgvq >= 1:
    val_text_loader = dataset_VQ.DATALoader(args.dataname,
                                        32,
                                        window_size=args.window_size,
                                        unit_length=2**args.down_t,
                                        val=True)
    # val_text_loader_iter = dataset_VQ.cycle(val_text_loader)
print("args = ",args)


##### ---- Optimizer & Scheduler ---- #####
optimizer = optim.AdamW(net.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_scheduler, gamma=args.gamma)
  

Loss = losses.ReConsLoss(args.recons_loss, args.nb_joints)

##### ------ warm-up ------- #####
avg_recons, avg_perplexity, avg_commit = 0., 0., 0.
avg_contrastive, avg_mlm = 0., 0.

for nb_iter in range(1, args.warm_up_iter):
    
    optimizer, current_lr = update_lr_warm_up(optimizer, nb_iter, args.warm_up_iter, args.lr)
    
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

    pred_motion, loss_commit, perplexity, loss_lgvq = net(gt_motion, gt_motion_mask, text_mask, name)
    loss_motion = Loss(pred_motion, gt_motion, gt_motion_mask)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion, gt_motion_mask)
    contrastive_loss, mlm_loss = loss_lgvq
    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel + args.loss_contrastive * contrastive_loss + args.loss_mlm * mlm_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_contrastive += contrastive_loss.item()
    avg_mlm += mlm_loss.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        avg_contrastive /= args.print_iter
        avg_mlm /= args.print_iter
        
        logger.info(f"Warmup. Iter {nb_iter} :  lr {current_lr:.5f} \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Contrastive. {avg_contrastive:.5f} \t Mlm. {avg_mlm:.5f}")
        
        avg_recons, avg_perplexity, avg_commit, avg_contrastive, avg_mlm = 0., 0., 0., 0., 0.

##### ---- Training ---- #####
avg_recons, avg_perplexity, avg_commit, avg_contrastive, avg_mlm = 0., 0., 0., 0., 0.
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, best_mpjpe = eval_trans.evaluation_vqvae(args.run_dir, val_loader, net, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, eval_wrapper=eval_wrapper, draw=False, best_mpjpe=100)
if args.lgvq >= 1:
    R1, R2 = eval_trans.evaluation_vqvae_text(val_text_loader, net)
if args.freeze_encdec != 0 and args.lgvq != 0:
    net = freeze_encdec(net)
for nb_iter in range(1, args.total_iter + 1):
    
    gt_motion = next(train_loader_iter)
    if isinstance(gt_motion, tuple) or isinstance(gt_motion, list):
        if len(gt_motion) == 2:
            gt_motion, gt_motion_mask = gt_motion
            text_mask = None
        elif len(gt_motion) == 3:
            gt_motion, gt_motion_mask, text_mask = gt_motion
        elif len(gt_motion) == 4:
            gt_motion, gt_motion_mask, text_mask, name = gt_motion
        elif len(gt_motion) == 5:
            gt_motion, gt_motion_mask, text_mask, name, text = gt_motion
    else:
        gt_motion_mask, text_mask = None, None
    gt_motion = gt_motion.cuda().float() # bs, nb_joints, joints_dim, seq_len
    if gt_motion_mask is not None:
        gt_motion_mask = gt_motion_mask.cuda().long() # bs, seq_len
    pred_motion, loss_commit, perplexity, loss_lgvq = net(gt_motion, gt_motion_mask, text_mask, name)
    contrastive_loss, mlm_loss = loss_lgvq
    loss_motion = Loss(pred_motion, gt_motion, gt_motion_mask)
    loss_vel = Loss.forward_vel(pred_motion, gt_motion, gt_motion_mask)
    
    loss = loss_motion + args.commit * loss_commit + args.loss_vel * loss_vel + args.loss_contrastive * contrastive_loss + args.loss_mlm * mlm_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()
    
    avg_recons += loss_motion.item()
    avg_perplexity += perplexity.item()
    avg_commit += loss_commit.item()
    avg_mlm += mlm_loss.item()
    avg_contrastive += contrastive_loss.item()
    
    if nb_iter % args.print_iter ==  0 :
        avg_recons /= args.print_iter
        avg_perplexity /= args.print_iter
        avg_commit /= args.print_iter
        avg_contrastive /= args.print_iter
        avg_mlm /= args.print_iter
        
        writer.add_scalar('./Train/L1', avg_recons, nb_iter)
        writer.add_scalar('./Train/PPL', avg_perplexity, nb_iter)
        writer.add_scalar('./Train/Commit', avg_commit, nb_iter)
        
        logger.info(f"Train. Iter {nb_iter} : \t Commit. {avg_commit:.5f} \t PPL. {avg_perplexity:.2f} \t Recons.  {avg_recons:.5f} \t Contrastive. {avg_contrastive:.5f} \t Mlm. {avg_mlm:.5f}")
        
        avg_recons, avg_perplexity, avg_commit, avg_contrastive, avg_mlm = 0., 0., 0., 0., 0.

    if nb_iter % args.eval_iter==0 :
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, best_mpjpe = eval_trans.evaluation_vqvae(args.run_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper=eval_wrapper, best_mpjpe=best_mpjpe)
        if args.freeze_encdec != 0 and args.lgvq != 0:
            net = freeze_encdec(net)
        