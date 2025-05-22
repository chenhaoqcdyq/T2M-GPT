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



if args_vq.lgvq == 1 and args.sample_way != 2:
    num_vq_trans = args.nb_code * 2 + 1
elif args_vq.lgvq == 1 and args.sample_way == 2:
    num_vq_trans = args.nb_code
else:
    num_vq_trans = args.nb_code
# 测试codebook的size是否能够对生成进行提点
if args.test_nb:
    num_vq_trans = args.nb_code * 2 + 1
print("num_vq_trans = ", num_vq_trans)
if args_vq.lgvq == 1 and args.sample_way == 0:
    semantic_flag = True
else:
    semantic_flag = False
if "down_vqvae" in args_vq:
    if args_vq.down_vqvae and args_vq.down_t == 2:
        unit_length = 4
    elif args_vq.down_vqvae and args_vq.down_t == 1:
        unit_length = 2
    elif args_vq.down_vqvae and args_vq.down_t == 3:
        unit_length = 8
    else:
        unit_length = 1
else:
    unit_length = 1
semantic_len = ((196 // unit_length) + 3) // 4 + 1
trans_encoder = trans.Text2Motion_Transformer(num_vq=num_vq_trans, 
                                embed_dim=args.embed_dim_gpt, 
                                clip_dim=args.clip_dim, 
                                block_size=args.block_size, 
                                num_layers=args.num_layers, 
                                n_head=args.n_head_gpt, 
                                drop_out_rate=args.drop_out_rate, 
                                fc_rate=args.ff_rate,
                                semantic_flag=semantic_flag,
                                semantic_len=semantic_len,
                                dual_head_flag=(args.sample_way == 2),
                                uncond_prob=args.classfg)


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
generate_idx = False
for batch in tqdm(train_loader_token):
    pose, name, text = batch
    # if os.path.exists(pjoin(args.vq_dir, name[0] +'.npy')):
    #     if args.lgvq:
    #         if os.path.exists(pjoin(args.vq_dir, name[0] +'_sem.npy')):
    #             continue
    #     else:
    #         continue
    if args.exp_name == 'GPT':
        if os.path.exists(pjoin(args.vq_dir, name[0] +'.npz')):
            continue
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
    if generate_idx == True:
        m_tokens_len = motion_idx.shape[1]
        max_motion_length = 197
        mot_end_idx = 512
        mot_pad_idx = 513
        m_tokens_sem = motion_idx[0]
        if m_tokens_len+1 < max_motion_length:
            m_tokens_result = np.concatenate([m_tokens_sem, np.ones((1), dtype=int) * mot_end_idx, np.ones((max_motion_length-1-m_tokens_len), dtype=int) * mot_pad_idx], axis=0)
        else:
            m_tokens_result = np.concatenate([m_tokens_sem[:max_motion_length-1], np.ones((1), dtype=int) * mot_end_idx], axis=0)
        np.savez(pjoin("/workspace/motion_diffusion/T2M-GPT/dataset/HumanML3D/T2M_with_end_val", name[0] +'.npz'), motion=m_tokens_result, text=text)
    else:
        if sem_idx is not None:
            sem_idx = sem_idx.cpu().numpy() # (1, x)
            np.savez(pjoin(args.vq_dir, name[0] +'.npz'), motion=motion_idx, sem=sem_idx, text=text)
        else:
            np.savez(pjoin(args.vq_dir, name[0] +'.npz'), motion=motion_idx, text=text)

if args_vq.lgvq == 1:
    from dataset import dataset_TM_train_sem as dataset_TM_train
    if 'sample_way' not in args:
        args.sample_way = 0
    train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, vq_name, unit_length=unit_length, sample_way=args.sample_way)
else:
    from dataset import dataset_TM_train
    train_loader = dataset_TM_train.DATALoader(args.dataname, args.batch_size, args.nb_code, vq_name, unit_length=unit_length, test_nb=args.test_nb if 'test_nb' in args else False)


train_loader_iter = dataset_TM_train.cycle(train_loader)

##### ---- Training ---- #####
# best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer_batch(args.out_dir, val_loader, net, trans_encoder, logger, writer, 0, best_fid=1000, best_iter=0, best_div=100, best_top1=0, best_top2=0, best_top3=0, best_matching=100, clip_model=clip_model, eval_wrapper=eval_wrapper, semantic_flag=((args_vq.lgvq==1 or args.test_nb) and args.sample_way != 2), draw=False, dual_head_flag=(args.sample_way == 2))
best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching = 1000, 0, 100, 0, 0, 0, 100
while nb_iter <= args.total_iter:
    
    batch = next(train_loader_iter)
    clip_text, m_tokens, m_tokens_len = batch
    if isinstance(m_tokens_len, list) and len(m_tokens_len) == 2:
        m_tokens_len, sem_tokens_len = m_tokens_len
        m_tokens_len, sem_tokens_len = m_tokens_len.cuda(), sem_tokens_len.cuda()
    else:
        m_tokens_len = m_tokens_len.cuda()
        sem_tokens_len = None
    m_tokens = m_tokens.cuda()
    bs = m_tokens.shape[0]
    target = m_tokens    # (bs, 26)
    target = target.cuda()
    # if args.classfg == 0:
    text = clip.tokenize(clip_text, truncate=True).cuda()
    # else:
        # # 生成mask来决定哪些样本的文本被替换为空
        # mask = torch.bernoulli(args.classfg * torch.ones(len(clip_text), device='cuda'))
        # # 创建新的文本列表，将被mask的文本替换为空字符串
        # masked_text = ["" if mask[i] else clip_text[i] for i in range(len(clip_text))]
        # text = clip.tokenize(masked_text, truncate=True).cuda()
        
    feat_clip_text = clip_model.encode_text(text).float()

    input_index = target[:,:-1]

    if args.pkeep == -1:
        proba = np.random.rand(1)[0]
        mask = torch.bernoulli(proba * torch.ones(input_index.shape,
                                                         device=input_index.device))
    else:
        mask = torch.bernoulli(args.pkeep * torch.ones(input_index.shape,
                                                         device=input_index.device))
    mask = mask.round().to(dtype=torch.int64) # torch.Size([128, 50])
    r_indices = torch.randint_like(input_index, args.nb_code)
    a_indices = mask*input_index+(1-mask)*r_indices
    if sem_tokens_len is not None:
        cls_pred = trans_encoder(a_indices, feat_clip_text, semantic_valid_lengths=sem_tokens_len) # torch.Size([128, 50]), torch.Size([128, 512])
    else:
        cls_pred = trans_encoder(a_indices, feat_clip_text) # torch.Size([128, 50]), torch.Size([128, 512])
    cls_pred = cls_pred.contiguous() # torch.Size([128, 51, 513])

    loss_cls = 0.0
    # loss_sem_cls = 0.0
    for i in range(bs):
        # loss function     (26), (26, 513)
        if sem_tokens_len is None:
            loss_cls += loss_ce(cls_pred[i][:m_tokens_len[i] + 1], target[i][:m_tokens_len[i] + 1]) / bs
            # Accuracy
            probs = torch.softmax(cls_pred[i][:m_tokens_len[i] + 1], dim=-1)
        else:
            pred_all = torch.cat([cls_pred[i][:sem_tokens_len[i] + 1], cls_pred[i][semantic_len:semantic_len + m_tokens_len[i] + 1]], dim=0)
            target_all = torch.cat([target[i][:sem_tokens_len[i] + 1], target[i][semantic_len:semantic_len + m_tokens_len[i] + 1]], dim=0)
            loss_cls += loss_ce(pred_all, target_all) / bs
            probs = torch.softmax(pred_all, dim=-1)
        

        if args.if_maxtest:
            _, cls_pred_index = torch.max(probs, dim=-1)

        else:
            dist = Categorical(probs)
            cls_pred_index = dist.sample()
        if sem_tokens_len is None:
            right_num += (cls_pred_index.flatten(0) == target[i][:m_tokens_len[i] + 1].flatten(0)).sum().item()
        else:
            right_num += (cls_pred_index.flatten(0) == target_all.flatten(0)).sum().item()

    ## global loss
    optimizer.zero_grad()
    loss_cls.backward()
    optimizer.step()
    scheduler.step()

    avg_loss_cls = avg_loss_cls + loss_cls.item()
    if sem_tokens_len is None:
        nb_sample_train = nb_sample_train + (m_tokens_len + 1).sum().item()
    else:
        nb_sample_train = nb_sample_train + (sem_tokens_len + 1 + m_tokens_len + 1).sum().item()

    nb_iter += 1
    if nb_iter % args.print_iter ==  0 :
        avg_loss_cls = avg_loss_cls / args.print_iter
        avg_acc = right_num * 100 / nb_sample_train
        writer.add_scalar('./Loss/train', avg_loss_cls, nb_iter)
        writer.add_scalar('./ACC/train', avg_acc, nb_iter)
        msg = f"Train. Iter {nb_iter} : Loss. {avg_loss_cls:.5f}, ACC. {avg_acc:.4f}"
        logger.info(msg)
        avg_loss_cls = 0.
        right_num = 0
        nb_sample_train = 0

    if nb_iter % args.eval_iter ==  0:
        best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger = eval_trans.evaluation_transformer_batch(args.out_dir, val_loader, net, trans_encoder, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model=clip_model, eval_wrapper=eval_wrapper, semantic_flag=((args_vq.lgvq==1 or args.test_nb) and args.sample_way != 2), draw=False, dual_head_flag=(args.sample_way == 2))

    if nb_iter == args.total_iter: 
        msg_final = f"Train. Iter {best_iter} : FID. {best_fid:.5f}, Diversity. {best_div:.4f}, TOP1. {best_top1:.4f}, TOP2. {best_top2:.4f}, TOP3. {best_top3:.4f}"
        logger.info(msg_final)
        break            