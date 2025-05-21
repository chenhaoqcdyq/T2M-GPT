import os

import clip
import numpy as np
import torch
from scipy import linalg
from tqdm import tqdm
import visualization.plot_3d_global as plot_3d
from utils.motion_process import recover_from_ric
from typing import Any, List, Tuple, Union

def tensorborad_add_video_xyz(writer, xyz, nb_iter, tag, nb_vis=4, title_batch=None, outname=None):
    xyz = xyz[:1]
    bs, seq = xyz.shape[:2]
    xyz = xyz.reshape(bs, seq, -1, 3)
    plot_xyz = plot_3d.draw_to_batch(xyz.cpu().numpy(),title_batch, outname)
    plot_xyz =np.transpose(plot_xyz, (0, 1, 4, 2, 3)) 
    writer.add_video(tag, plot_xyz, nb_iter, fps = 20)

def calculate_mpjpe(gt_joints, pred_joints):
    """
    gt_joints: num_poses x num_joints(22) x 3
    pred_joints: num_poses x num_joints(22) x 3
    (obtained from recover_from_ric())
    """
    assert gt_joints.shape == pred_joints.shape, f"GT shape: {gt_joints.shape}, pred shape: {pred_joints.shape}"

    # Align by root (pelvis)
    pelvis = gt_joints[:, [0]].mean(1)
    gt_joints = gt_joints - torch.unsqueeze(pelvis, dim=1)
    pelvis = pred_joints[:, [0]].mean(1)
    pred_joints = pred_joints - torch.unsqueeze(pelvis, dim=1)

    # Compute MPJPE
    mpjpe = torch.linalg.norm(pred_joints - gt_joints, dim=-1) # num_poses x num_joints=22
    mpjpe_seq = mpjpe.mean(-1) # num_poses

    return mpjpe_seq

@torch.no_grad()   
def evaluation_vqvae_text(val_loader, net):
    net.eval()
    R1 = []
    R2 = []
    for batch in tqdm(val_loader):
        gt_motion = batch
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
        gt_motion = gt_motion.cuda().float()
        gt_motion_mask = gt_motion_mask.cuda().long()
        with torch.no_grad():
            result = net.text_motion_topk(gt_motion, motion_mask=gt_motion_mask, topk=5, text_mask=text_mask, text=text)
        global_R, pred_R = result
        R1.append(global_R)
        R2.append(pred_R)
        print(result)
    R1_mean = np.mean(np.array(R1), axis=0)
    R2_mean = np.mean(np.array(R2), axis=0)
    return R1_mean, R2_mean

@torch.no_grad()        
def evaluation_vqvae(out_dir, val_loader, net, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, best_mpjpe=100) : 
    net.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []


    motion_annotation_list = []
    motion_pred_list = []

    R_precision_real = 0
    R_precision = 0

    nb_sample = 0
    matching_score_real = 0
    matching_score_pred = 0
    mpjpe = 0
    num_poses = 0
    for batch in val_loader:
        word_embeddings, pos_one_hots, caption, sent_len, motion, m_length, token, name = batch

        motion = motion.cuda()
        et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, motion, m_length)
        bs, seq = motion.shape[0], motion.shape[1]

        num_joints = 21 if motion.shape[-1] == 251 else 22
        
        pred_pose_eval = torch.zeros((bs, seq, motion.shape[-1])).cuda()

        for i in range(bs):
            pose = val_loader.dataset.inv_transform(motion[i:i+1, :m_length[i], :].detach().cpu().numpy())
            pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

            result = net(motion[i:i+1, :m_length[i]])
            pred_pose = result[0][:,0:m_length[i],:]
            # pred_pose, loss_commit, perplexity = net(motion[i:i+1, :m_length[i]])
            pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
            pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
            
            if savenpy:
                np.save(os.path.join(out_dir, name[i]+'_gt.npy'), pose_xyz[:, :m_length[i]].cpu().numpy())
                np.save(os.path.join(out_dir, name[i]+'_pred.npy'), pred_xyz.detach().cpu().numpy())
            pred_pose_eval[i:i+1,:m_length[i],:] = pred_pose
            mpjpe += torch.sum(calculate_mpjpe(pose_xyz, pred_xyz))
            num_poses += pose_xyz.shape[0]
            if i < min(4, bs):
                draw_org.append(pose_xyz)
                draw_pred.append(pred_xyz)
                draw_text.append(caption[i])

        et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, m_length)

        motion_pred_list.append(em_pred)
        motion_annotation_list.append(em)
            
        temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
        R_precision_real += temp_R
        matching_score_real += temp_match
        temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
        R_precision += temp_R
        matching_score_pred += temp_match

        nb_sample += bs
    mpjpe = mpjpe / num_poses
    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, mpjpe. {mpjpe:.4f}"
    logger.info(msg)
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 5000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)   

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))
    
    if mpjpe < best_mpjpe : 
        msg = f"--> --> \t mpjpe Improved from {best_mpjpe:.5f} to {mpjpe:.5f} !!!"
        logger.info(msg)
        best_mpjpe = mpjpe
        if save:
            torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_best_mpjpe.pth'))

    if save:
        torch.save({'net' : net.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    net.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger, best_mpjpe


@torch.no_grad()        
def evaluation_transformer(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, draw = True, save = True, savegif=False, semantic_flag=False, dual_head_flag=False) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in tqdm(val_loader):
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22
            
            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            for k in range(bs):
                try:
                    index_motion = trans.sample(feat_clip_text[k:k+1], False)
                except:
                    index_motion = torch.ones(1,4).cuda().long()
                if semantic_flag:
                    index_motion = index_motion[index_motion >= 513] - 513
                    if index_motion.shape[0] == 0:
                        index_motion = torch.ones(1,4).cuda().long()
                    pred_pose = net.forward_decoder(index_motion)
                elif dual_head_flag:
                    index_motion = index_motion[..., trans.semantic_len:]
                    if index_motion.shape[0] == 0:
                        index_motion = torch.ones(1,4).cuda().long()
                    pred_pose = net.forward_decoder(index_motion)
                else:
                    if index_motion.shape[0] == 0:
                        index_motion = torch.ones(1,4).cuda().long()
                    pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if i == 0 and k < 4:
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)
    
    
    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)

    
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)
            
        if nb_iter % 10000 == 0 : 
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)

    
    if fid < best_fid : 
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))
    
    if matching_score_pred < best_matching : 
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) : 
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity

    if R_precision[0] > best_top1 : 
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]

    if R_precision[1] > best_top2 : 
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]
    
    if R_precision[2] > best_top3 : 
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger

@torch.no_grad()
def evaluation_transformer_batch(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, clip_model, eval_wrapper, draw = True, save = True, savegif=False, semantic_flag=False, dual_head_flag=False) :
    """
    This is used for evaluate GPT at training stage.
    It excludes the multi-modality evaluation by simply set a circle only at 1 time.
    """
    trans.eval()
    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []

    motion_annotation_list = []
    motion_pred_list = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    for i in range(1):
        for batch in tqdm(val_loader):
            
            word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch

            bs, seq = pose.shape[:2]
            num_joints = 21 if pose.shape[-1] == 251 else 22

            text = clip.tokenize(clip_text, truncate=True).cuda()

            feat_clip_text = clip_model.encode_text(text).float()  # (B, 512)
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()


            # [Text-to-motion Generation] get generated parts' token sequence
            # get parts_index_motion given the feat_clip_text
            batch_parts_index_motion = trans.sample_batch(feat_clip_text, False)  # torch.Tensor: (B, seq_len)
            if dual_head_flag:
                batch_parts_index_motion = batch_parts_index_motion[..., trans.semantic_len:]
            max_motion_seq_len = batch_parts_index_motion.shape[1]

            for k in range(bs):

                parts_index_motion = []

                # get one sample
                part_index = batch_parts_index_motion[k:k+1]  # (1, seq_len)

                # find the earliest end token position
                idx = torch.nonzero(part_index == trans.num_vq)

                if idx.numel() == 0:
                    motion_seq_len = max_motion_seq_len
                else:
                    min_end_idx = idx[:,1].min()
                    motion_seq_len = min_end_idx

                # Truncate
                if motion_seq_len == 0:
                    # assign a nonsense motion index to handle length is 0 issue.
                    parts_index_motion = torch.ones(1, 4).cuda().long()  # (B, seq_len) B==1, seq_len==1
                else:
                    parts_index_motion = part_index[:,:motion_seq_len]



                '''
                index_motion: (B, nframes). Here: B == 1, nframes == predicted_length
                '''

                # [Token-to-RawMotion with VQ-VAE decoder] get each parts' raw motion
                parts_pred_pose = net.forward_decoder(parts_index_motion)  # (B, pred_nframes, parts_sk_dim)
                #   todo: support different shared_joint_rec_mode in the parts2whole function
                # pred_pose = val_loader.dataset.parts2whole(parts_pred_pose, mode=val_loader.dataset.dataset_name)  # (B, pred_nframes, raw_motion_dim)

                pred_pose = parts_pred_pose
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)  # save the min len

                # It's actually should use pred_len[k] to replace cur_len and seq for understanding convenience
                #   Below code seems equal to use pred_len[k].
                #   But should not change it to keep the same test code with T2M-GPT.
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if draw:
                    if i == 0 and k < 4:
                        pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                        pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)
                        draw_pred.append(pred_xyz)
                        draw_text_pred.append(clip_text[k])


            et_pred, em_pred = eval_wrapper.get_co_embeddings(
                word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)


                    for j in range(min(4, bs)):
                        draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                        draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs


    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov = calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample


    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}"
    logger.info(msg)


    if draw:
        writer.add_scalar('./Test/FID', fid, nb_iter)
        writer.add_scalar('./Test/Diversity', diversity, nb_iter)
        writer.add_scalar('./Test/top1', R_precision[0], nb_iter)
        writer.add_scalar('./Test/top2', R_precision[1], nb_iter)
        writer.add_scalar('./Test/top3', R_precision[2], nb_iter)
        writer.add_scalar('./Test/matching_score', matching_score_pred, nb_iter)


        if nb_iter % 10000 == 0 :
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/org_eval'+str(ii), nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, 'gt'+str(ii)+'.gif')] if savegif else None)

        if nb_iter % 10000 == 0 :
            for ii in range(4):
                tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/pred_eval'+str(ii), nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, 'pred'+str(ii)+'.gif')] if savegif else None)


    if fid < best_fid :
        msg = f"--> --> \t FID Improved from {best_fid:.5f} to {fid:.5f} !!!"
        logger.info(msg)
        best_fid, best_iter = fid, nb_iter
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_fid.pth'))

    if matching_score_pred < best_matching :
        msg = f"--> --> \t matching_score Improved from {best_matching:.5f} to {matching_score_pred:.5f} !!!"
        logger.info(msg)
        best_matching = matching_score_pred
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_matching.pth'))

    if abs(diversity_real - diversity) < abs(diversity_real - best_div) :
        msg = f"--> --> \t Diversity Improved from {best_div:.5f} to {diversity:.5f} !!!"
        logger.info(msg)
        best_div = diversity
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_div.pth'))

    if R_precision[0] > best_top1 :
        msg = f"--> --> \t Top1 Improved from {best_top1:.4f} to {R_precision[0]:.4f} !!!"
        logger.info(msg)
        best_top1 = R_precision[0]
        if save:
            torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_best_top1.pth'))

    if R_precision[1] > best_top2 :
        msg = f"--> --> \t Top2 Improved from {best_top2:.4f} to {R_precision[1]:.4f} !!!"
        logger.info(msg)
        best_top2 = R_precision[1]

    if R_precision[2] > best_top3 :
        msg = f"--> --> \t Top3 Improved from {best_top3:.4f} to {R_precision[2]:.4f} !!!"
        logger.info(msg)
        best_top3 = R_precision[2]

    if save:
        torch.save({'trans' : trans.state_dict()}, os.path.join(out_dir, 'net_last.pth'))

    trans.train()
    return best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, writer, logger

@torch.no_grad()        
def evaluation_transformer_test(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, semantic_flag=False, dual_head_flag=False, sample_cfg=False, mmod_gen_times=30) : 

    trans.eval()
    nb_sample = 0
    
    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in tqdm(val_loader):

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        if sample_cfg:
            uncond_feat = clip_model.encode_text(clip.tokenize('', truncate=True).cuda()).float()
            
            # cond_feat = clip_model.encode_text(clip_text[1:2]).float()
        motion_multimodality_batch = []
        for i in range(mmod_gen_times):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            
            for k in range(1):
                try:
                    if sample_cfg:
                        text_cond_uncond = torch.cat([feat_clip_text[k:k+1], uncond_feat.unsqueeze(0)], dim=0)
                        index_motion = trans.sample_cfg(text_cond_uncond, True)
                    else:
                        index_motion = trans.sample(feat_clip_text[k:k+1], False)
                except:
                    index_motion = torch.ones(1,4).cuda().long()
                if semantic_flag:
                    index_motion = index_motion[index_motion >= 513] - 513
                    if index_motion.shape[0] == 0:
                        index_motion = torch.ones(1,4).cuda().long()
                    pred_pose = net.forward_decoder(index_motion)
                elif dual_head_flag:
                    index_motion = index_motion[..., trans.semantic_len:]
                    if index_motion.shape[0] == 0:
                        index_motion = torch.ones(1,4).cuda().long()
                    pred_pose = net.forward_decoder(index_motion)
                else:
                    if index_motion.shape[0] == 0:
                        index_motion = torch.ones(1,4).cuda().long()
                    pred_pose = net.forward_decoder(index_motion)
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                # cur_len = pred_pose.shape[1]

                # pred_len[k] = min(cur_len, seq)
                # pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)
            if torch.isnan(em_pred).any():
                print(em_pred)
            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                if torch.isnan(em).any():
                    print(em)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    if mmod_gen_times < 10:
        multimodality = 0
    else:
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)
    
    
    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)
        
            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

@torch.no_grad()
def evaluation_transformer_test_batch(out_dir, val_loader, net, trans, logger, writer, nb_iter, best_fid, best_iter, best_div, best_top1, best_top2, best_top3, best_matching, best_multi, clip_model, eval_wrapper, draw = True, save = True, savegif=False, savenpy=False, mmod_gen_times=30, skip_mmod=False, dual_head_flag=False):

    trans.eval()

    if skip_mmod:
        mmod_gen_times = 1

    nb_sample = 0

    draw_org = []
    draw_pred = []
    draw_text = []
    draw_text_pred = []
    draw_name = []

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0

    for batch in tqdm(val_loader):

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22

        text = clip.tokenize(clip_text, truncate=True).cuda()

        feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(mmod_gen_times):  # mmod_gen_times default: 30
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()

            # [Text-to-motion Generation] get generated parts' token sequence
            # get parts_index_motion given the feat_clip_text
            batch_parts_index_motion = trans.sample_batch(feat_clip_text, False)  # List: [(B, seq_len), ..., (B, seq_len)]
            if dual_head_flag:
                batch_parts_index_motion = batch_parts_index_motion[..., trans.semantic_len:]

            max_motion_seq_len = batch_parts_index_motion.shape[1]


            for k in range(bs):

                min_motion_seq_len = max_motion_seq_len
                # parts_index_motion = []
                # for part_index, name in zip(batch_parts_index_motion, ['Root', 'R_Leg', 'L_Leg', 'Backbone', 'R_Arm', 'L_Arm']):

                # get one sample
                part_index = batch_parts_index_motion[k:k+1]  # (1, seq_len)

                # find the earliest end token position
                idx = torch.nonzero(part_index == trans.num_vq)

                # # Debug
                # print('part_index:', part_index)
                # print('nonzero_idx', idx)

                if idx.numel() == 0:
                    motion_seq_len = max_motion_seq_len
                else:
                    min_end_idx = idx[:,1].min()
                    motion_seq_len = min_end_idx

                # if motion_seq_len < min_motion_seq_len:
                min_motion_seq_len = motion_seq_len

                # parts_index_motion.append(part_index)

                # Truncate
                # for j in range(len(parts_index_motion)):
                if min_motion_seq_len == 0:
                    parts_index_motion = torch.ones(1,4).cuda().long()  # (B, seq_len) B==1, seq_len==1
                elif min_motion_seq_len <= 3:
                    # assign a nonsense motion index to handle length is 0 issue.
                    parts_index_motion = torch.cat([part_index[:,:min_motion_seq_len], torch.ones(1,4-min_motion_seq_len).cuda().long()], dim=1)
                else:
                    parts_index_motion = part_index[:,:min_motion_seq_len]



                # [Token-to-RawMotion with VQ-VAE decoder] get each parts' raw motion
                parts_pred_pose = net.forward_decoder(parts_index_motion)  # (B, pred_nframes, parts_sk_dim)
                #   todo: support different shared_joint_rec_mode in the parts2whole function
                # pred_pose = val_loader.dataset.parts2whole(parts_pred_pose, mode=val_loader.dataset.dataset_name)  # (B, pred_nframes, raw_motion_dim)
                pred_pose = parts_pred_pose
                cur_len = pred_pose.shape[1]

                pred_len[k] = min(cur_len, seq)
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]

                if i == 0 and (draw or savenpy):
                    pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                    pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                    if savenpy:
                        np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())

                    if draw:
                        if i == 0:
                            draw_pred.append(pred_xyz)
                            draw_text_pred.append(clip_text[k])
                            draw_name.append(name[k])

            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))

            if i == 0:
                pose = pose.cuda().float()

                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                if draw or savenpy:
                    pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                    pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                    if savenpy:
                        for j in range(bs):
                            np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())

                    if draw:
                        for j in range(bs):
                            draw_org.append(pose_xyz[j][:m_length[j]].unsqueeze(0))
                            draw_text.append(clip_text[j])

                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    if not skip_mmod:
        print('Calculate multimodality...')
        motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
        multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter {nb_iter} :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    logger.info(msg)


    if draw:
        for ii in range(len(draw_org)):
            tensorborad_add_video_xyz(writer, draw_org[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_org', nb_vis=1, title_batch=[draw_text[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_gt.gif')] if savegif else None)

            tensorborad_add_video_xyz(writer, draw_pred[ii], nb_iter, tag='./Vis/'+draw_name[ii]+'_pred', nb_vis=1, title_batch=[draw_text_pred[ii]], outname=[os.path.join(out_dir, draw_name[ii]+'_skel_pred.gif')] if savegif else None)

    trans.train()
    return fid, best_iter, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality, writer, logger

# (X - X_train)*(X - X_train) = -2X*X_train + X*X + X_train*X_train
def euclidean_distance_matrix(matrix1, matrix2):
    """
        Params:
        -- matrix1: N1 x D
        -- matrix2: N2 x D
        Returns:
        -- dist: N1 x N2
        dist[i, j] == distance(matrix1[i], matrix2[j])
    """
    assert matrix1.shape[1] == matrix2.shape[1]
    d1 = -2 * np.dot(matrix1, matrix2.T)    # shape (num_test, num_train)
    d2 = np.sum(np.square(matrix1), axis=1, keepdims=True)    # shape (num_test, 1)
    d3 = np.sum(np.square(matrix2), axis=1)     # shape (num_train, )
    dists = np.sqrt(d1 + d2 + d3)  # broadcasting
    return dists



def calculate_top_k(mat, top_k):
    size = mat.shape[0]
    gt_mat = np.expand_dims(np.arange(size), 1).repeat(size, 1)
    bool_mat = (mat == gt_mat)
    correct_vec = False
    top_k_list = []
    for i in range(top_k):
#         print(correct_vec, bool_mat[:, i])
        correct_vec = (correct_vec | bool_mat[:, i])
        # print(correct_vec)
        top_k_list.append(correct_vec[:, None])
    top_k_mat = np.concatenate(top_k_list, axis=1)
    return top_k_mat


def calculate_R_precision(embedding1, embedding2, top_k, sum_all=False):
    dist_mat = euclidean_distance_matrix(embedding1, embedding2)
    matching_score = dist_mat.trace()
    argmax = np.argsort(dist_mat, axis=1)
    top_k_mat = calculate_top_k(argmax, top_k)
    if sum_all:
        return top_k_mat.sum(axis=0), matching_score
    else:
        return top_k_mat, matching_score

def calculate_multimodality(activation, multimodality_times):
    assert len(activation.shape) == 3
    assert activation.shape[1] > multimodality_times
    num_per_sent = activation.shape[1]

    first_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    second_dices = np.random.choice(num_per_sent, multimodality_times, replace=False)
    dist = linalg.norm(activation[:, first_dices] - activation[:, second_dices], axis=2)
    return dist.mean()


def calculate_diversity(activation, diversity_times):
    assert len(activation.shape) == 2
    assert activation.shape[0] > diversity_times
    num_samples = activation.shape[0]

    first_indices = np.random.choice(num_samples, diversity_times, replace=False)
    second_indices = np.random.choice(num_samples, diversity_times, replace=False)
    dist = linalg.norm(activation[first_indices] - activation[second_indices], axis=1)
    return dist.mean()



def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)



def calculate_activation_statistics(activations):

    mu = np.mean(activations, axis=0)
    cov = np.cov(activations, rowvar=False)
    return mu, cov


def calculate_frechet_feature_distance(feature_list1, feature_list2):
    feature_list1 = np.stack(feature_list1)
    feature_list2 = np.stack(feature_list2)

    # normalize the scale
    mean = np.mean(feature_list1, axis=0)
    std = np.std(feature_list1, axis=0) + 1e-10
    feature_list1 = (feature_list1 - mean) / std
    feature_list2 = (feature_list2 - mean) / std

    dist = calculate_frechet_distance(
        mu1=np.mean(feature_list1, axis=0), 
        sigma1=np.cov(feature_list1, rowvar=False),
        mu2=np.mean(feature_list2, axis=0), 
        sigma2=np.cov(feature_list2, rowvar=False),
    )
    return dist

@torch.no_grad()        
def evaluation_transformer_test_vqvae(val_loader, net): 
    net.eval()
    nb_sample = 0

    motion_annotation_list = []
    motion_pred_list = []
    motion_multimodality = []
    R_precision_real = 0
    R_precision = 0
    matching_score_real = 0
    matching_score_pred = 0

    nb_sample = 0
    
    for batch in tqdm(val_loader):

        word_embeddings, pos_one_hots, clip_text, sent_len, pose, m_length, token, name = batch
        bs, seq = pose.shape[:2]
        num_joints = 21 if pose.shape[-1] == 251 else 22
        
        # text = clip.tokenize(clip_text, truncate=True).cuda()

        # feat_clip_text = clip_model.encode_text(text).float()
        motion_multimodality_batch = []
        for i in range(30):
            pred_pose_eval = torch.zeros((bs, seq, pose.shape[-1])).cuda()
            pred_len = torch.ones(bs).long()
            motion_mask = torch.ones(bs, seq).cuda().long()
            for k in range(bs):
                motion_mask[k, m_length[k]:] = 0
            pred_pose, _, _, _ = net.encode(pose.cuda(), motion_mask)
            for k in range(bs):
                # try:
                #     motion_mask = 
                #     index_motion = net(pose[k:k+1], True)
                #     # index_motion = trans.sample(feat_clip_text[k:k+1], True)
                # except:
                #     index_motion = torch.ones(1,1).cuda().long()

                # pred_pose = net.forward_decoder(index_motion)
                # cur_len = pred_pose.shape[1]
                cur_len = m_length[k]

                pred_len[k] = min(cur_len, seq)
                # pred_pose_eval[k:k+1, :cur_len] = pred_pose[:, :seq]
                pred_pose_eval[k:k+1, :cur_len] = pred_pose[k:k+1, :cur_len]

                # if i == 0 and (draw or savenpy):
                #     pred_denorm = val_loader.dataset.inv_transform(pred_pose.detach().cpu().numpy())
                #     pred_xyz = recover_from_ric(torch.from_numpy(pred_denorm).float().cuda(), num_joints)

                #     if savenpy:
                #         np.save(os.path.join(out_dir, name[k]+'_pred.npy'), pred_xyz.detach().cpu().numpy())


            et_pred, em_pred = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pred_pose_eval, pred_len)

            motion_multimodality_batch.append(em_pred.reshape(bs, 1, -1))
            
            if i == 0:
                pose = pose.cuda().float()
                
                et, em = eval_wrapper.get_co_embeddings(word_embeddings, pos_one_hots, sent_len, pose, m_length)
                motion_annotation_list.append(em)
                motion_pred_list.append(em_pred)

                # if draw or savenpy:
                #     pose = val_loader.dataset.inv_transform(pose.detach().cpu().numpy())
                #     pose_xyz = recover_from_ric(torch.from_numpy(pose).float().cuda(), num_joints)

                #     if savenpy:
                #         for j in range(bs):
                #             np.save(os.path.join(out_dir, name[j]+'_gt.npy'), pose_xyz[j][:m_length[j]].unsqueeze(0).cpu().numpy())


                temp_R, temp_match = calculate_R_precision(et.cpu().numpy(), em.cpu().numpy(), top_k=3, sum_all=True)
                R_precision_real += temp_R
                matching_score_real += temp_match
                temp_R, temp_match = calculate_R_precision(et_pred.cpu().numpy(), em_pred.cpu().numpy(), top_k=3, sum_all=True)
                R_precision += temp_R
                matching_score_pred += temp_match

                nb_sample += bs

        motion_multimodality.append(torch.cat(motion_multimodality_batch, dim=1))

    motion_annotation_np = torch.cat(motion_annotation_list, dim=0).cpu().numpy()
    motion_pred_np = torch.cat(motion_pred_list, dim=0).cpu().numpy()
    gt_mu, gt_cov  = calculate_activation_statistics(motion_annotation_np)
    mu, cov= calculate_activation_statistics(motion_pred_np)

    diversity_real = calculate_diversity(motion_annotation_np, 300 if nb_sample > 300 else 100)
    diversity = calculate_diversity(motion_pred_np, 300 if nb_sample > 300 else 100)

    R_precision_real = R_precision_real / nb_sample
    R_precision = R_precision / nb_sample

    matching_score_real = matching_score_real / nb_sample
    matching_score_pred = matching_score_pred / nb_sample

    multimodality = 0
    motion_multimodality = torch.cat(motion_multimodality, dim=0).cpu().numpy()
    multimodality = calculate_multimodality(motion_multimodality, 10)

    fid = calculate_frechet_distance(gt_mu, gt_cov, mu, cov)

    msg = f"--> \t Eva. Iter :, FID. {fid:.4f}, Diversity Real. {diversity_real:.4f}, Diversity. {diversity:.4f}, R_precision_real. {R_precision_real}, R_precision. {R_precision}, matching_score_real. {matching_score_real}, matching_score_pred. {matching_score_pred}, multimodality. {multimodality:.4f}"
    # logger.info(msg)
    print(msg)
    


    # trans.train()
    return fid, diversity, R_precision[0], R_precision[1], R_precision[2], matching_score_pred, multimodality

class EasyDict(dict):
    """Convenience class that behaves like a dict but allows access with the attribute syntax. From stylegan2-ADA"""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        del self[name]

if __name__ == "__main__":
    import os
    import re 
    import torch
    import numpy as np

    from torch.utils.tensorboard import SummaryWriter
    from os.path import join as pjoin
    # from torch.distributions import Categorical
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
    # args = option_trans.get_args_parser()
    # vqvae_train_dir = "output/VQVAE_all_motion_best"
    # training_options_path = os.path.join(vqvae_train_dir, 'train_config.json')
    # with open(training_options_path, 'r') as f:
    #     train_args_dict = json.load(f)  # dict
    # train_args = EasyDict(train_args_dict)  # convert dict to easydict for convenience
    # args = train_args
    resume_pth = "output/00131-t2m-VQVAE_all_motion/VQVAE-VQVAE_all_motion-t2m/net_last.pth"
    path = os.path.dirname(resume_pth)
    json_file = os.path.join(path, 'train_config.json')
    with open(json_file, 'r') as f:
        train_args_dict = json.load(f)  # dict
    args = eval_trans.EasyDict(train_args_dict) 
    torch.manual_seed(args.seed)
    ##### ---- Dataloader ---- #####
    train_loader_token = dataset_tokenize.DATALoader(args.dataname, 1, unit_length=1)

    from utils.word_vectorizer import WordVectorizer
    w_vectorizer = WordVectorizer('./glove', 'our_vab')
    val_loader = dataset_TM_eval.DATALoader(args.dataname, False, 32, w_vectorizer)

    dataset_opt_path = 'checkpoints/kit/Comp_v6_KLD005/opt.txt' if args.dataname == 'kit' else 'checkpoints/t2m/Comp_v6_KLD005/opt.txt'

    wrapper_opt = get_opt(dataset_opt_path, torch.device('cuda'))
    eval_wrapper = EvaluatorModelWrapper(wrapper_opt)

    ##### ---- Network ---- #####
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=torch.device('cuda'), jit=False)  # Must set jit=False for training
    clip.model.convert_weights(clip_model)  # Actually this line is unnecessary since clip by default already on float16
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

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
                        #    args.vq_norm,
                        enc=args.enc,
                        lgvq=args.lgvq,
                        causal=args.causal)

    # if args.lgvq:
    #     num_vq_trans = args.nb_code * 2 + 1
    # else:
    #     num_vq_trans = args.nb_code
    # trans_encoder = trans.Text2Motion_Transformer(num_vq=num_vq_trans, 
    #                                 embed_dim=args.embed_dim_gpt, 
    #                                 clip_dim=args.clip_dim, 
    #                                 block_size=args.block_size, 
    #                                 num_layers=args.num_layers, 
    #                                 n_head=args.n_head_gpt, 
    #                                 drop_out_rate=args.drop_out_rate, 
    #                                 fc_rate=args.ff_rate)


    print ('loading checkpoint from {}'.format(resume_pth))
    ckpt = torch.load(resume_pth, map_location='cpu')
    net.load_state_dict(ckpt['net'], strict=True)
    net.eval()
    net.cuda()

    # if args.resume_trans is not None:
    #     print ('loading transformer checkpoint from {}'.format(args.resume_trans))
    #     ckpt = torch.load(args.resume_trans, map_location='cpu')
    #     trans_encoder.load_state_dict(ckpt['trans'], strict=True)
    # trans_encoder.train()
    # trans_encoder.cuda()
    
    evaluation_transformer_test_vqvae(val_loader, net)