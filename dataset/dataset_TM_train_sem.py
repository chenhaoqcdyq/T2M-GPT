import os
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate


def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, sem_codebook_size=512, down_sample = True, sample_way = 0):
        self.down_sample = down_sample
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.sem_start_idx = 0
        if sample_way == 2:
            self.sem_end_idx = sem_codebook_size # 512
            self.sem_pad_idx = self.sem_end_idx + 1 # 513
            self.start_motion_idx = 0
            self.mot_end_idx = codebook_size + self.start_motion_idx # 512
            self.mot_pad_idx = self.mot_end_idx + 1 # 513
        else:
            self.sem_end_idx = sem_codebook_size
            self.start_motion_idx = self.sem_end_idx + 1
            self.mot_end_idx = codebook_size + self.start_motion_idx
            self.mot_pad_idx = self.mot_end_idx + 1
        
        self.sample_way = sample_way
        
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            
            kinematic_chain = paramUtil.kit_kinematic_chain

        split_file = pjoin(self.data_root, 'train.txt')
        self.max_motion_length = 196 // unit_length + 4 # 4个特殊符号
        if self.down_sample:
            self.max_motion_length = self.max_motion_length + (self.max_motion_length - 4) // 4
        # if self.sample_way == 2:
        self.sem_max_length = ((196 // unit_length) + 3) // 4 + 1

        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        new_name_list = []
        data_dict = {}
        for name in tqdm(id_list):
            try:
                if os.path.exists(pjoin(self.data_root, tokenizer_name, '%s.npz'%name)):
                    data = np.load(pjoin(self.data_root, tokenizer_name, '%s.npz'%name))
                    m_token_list = data['motion']
                    sem_token_list = data['sem']
                else:
                    m_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s.npy'%name))
                    sem_token_list = np.load(pjoin(self.data_root, tokenizer_name, '%s_sem.npy'%name))
                # Read text
                with cs.open(pjoin(self.text_dir, name + '.txt')) as f:
                    text_data = []
                    flag = False
                    lines = f.readlines()

                    for line in lines:
                        try:
                            text_dict = {}
                            line_split = line.strip().split('#')
                            caption = line_split[0]
                            t_tokens = line_split[1].split(' ')
                            f_tag = float(line_split[2])
                            to_tag = float(line_split[3])
                            f_tag = 0.0 if np.isnan(f_tag) else f_tag
                            to_tag = 0.0 if np.isnan(to_tag) else to_tag

                            text_dict['caption'] = caption
                            text_dict['tokens'] = t_tokens
                            if f_tag == 0.0 and to_tag == 0.0:
                                flag = True
                                text_data.append(text_dict)
                            else:
                                m_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in m_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
                                # sem_token_list_new = [tokens[int(f_tag*fps/unit_length) : int(to_tag*fps/unit_length)] for tokens in sem_token_list if int(f_tag*fps/unit_length) < int(to_tag*fps/unit_length)]
                                sem_token_list_new = sem_token_list
                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                       'sem_token_list': sem_token_list_new,
                                                       'text':[text_dict]}
                                new_name_list.append(new_name)
                        except:
                            pass

                if flag:
                    data_dict[name] = {'m_token_list': m_token_list,
                                       'sem_token_list': sem_token_list,
                                       'text':text_data}
                    new_name_list.append(name)
            except:
                pass
        self.data_dict = data_dict
        self.name_list = new_name_list

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list, sem_token_list = data['m_token_list'], data['text'], data['sem_token_list']
        m_tokens = random.choice(m_token_list)
        sem_tokens = random.choice(sem_token_list)
        text_data = random.choice(text_list)
        caption= text_data['caption']

        if self.sample_way == 0:
            m_tokens = m_tokens + self.start_motion_idx
            m_tokens_sem = np.concatenate([sem_tokens, self.sem_end_idx * np.ones((1), dtype=int), m_tokens], axis=0)
            m_tokens_len = m_tokens_sem.shape[0]
            if m_tokens_len+1 < self.max_motion_length:
                m_tokens_result = np.concatenate([m_tokens_sem, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
            else:
                m_tokens_result = np.concatenate([m_tokens_sem, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)
        elif self.sample_way == 1:
            # 确保sem_tokens长度是m_tokens的1/4
            # sem_len = len(m_tokens) // 4
            # sem_tokens = sem_tokens[:sem_len]
            
            # 创建穿插序列
            m_tokens = m_tokens + self.start_motion_idx
            
            # 按照模式穿插排列
            m_tokens_sem = []
            for i in range(0, len(m_tokens)):
                if i % 4 == 0:
                    # 添加sem_tokens
                    m_tokens_sem.append(sem_tokens[i // 4])
                m_tokens_sem.append(m_tokens[i])

            
            m_tokens_sem = np.array(m_tokens_sem)
            m_tokens_len = m_tokens_sem.shape[0]
            
            # 添加end token和padding
            if m_tokens_len+1 < self.max_motion_length:
                m_tokens_result = np.concatenate([m_tokens_sem, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
            else:
                m_tokens_result = np.concatenate([m_tokens_sem, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)
        elif self.sample_way == 2:
            # 1. Process Semantic Tokens
            current_sem_tokens_list = list(sem_tokens) 
            
            if len(current_sem_tokens_list) >= self.sem_max_length:
                # Truncate and add end token
                processed_sem_part_list = current_sem_tokens_list[:self.sem_max_length-1] + [self.sem_end_idx]
            else:
                # Add end token and pad
                processed_sem_part_list = current_sem_tokens_list + [self.sem_end_idx]
                sem_padding_count = self.sem_max_length - len(processed_sem_part_list)
                if sem_padding_count > 0:
                    processed_sem_part_list.extend([self.sem_pad_idx] * sem_padding_count)
            
            processed_sem_tokens_np = np.array(processed_sem_part_list, dtype=int)

            # 2. Process Motion Tokens
            motion_tokens_adjusted = m_tokens + self.start_motion_idx

            # 3. Combine semantic and motion tokens, then add motion end token
            combined_tokens_np = np.concatenate((processed_sem_tokens_np.reshape(-1), motion_tokens_adjusted.reshape(-1)))
            
            final_tokens_list = list(combined_tokens_np)
            final_tokens_list.append(self.mot_end_idx)

            # 4. Pad or Truncate to self.max_motion_length
            current_length_with_mot_end = len(final_tokens_list)

            if current_length_with_mot_end < self.max_motion_length:
                mot_padding_count = self.max_motion_length - current_length_with_mot_end
                final_tokens_list.extend([self.mot_pad_idx] * mot_padding_count)
            elif current_length_with_mot_end > self.max_motion_length:
                final_tokens_list = final_tokens_list[:self.max_motion_length-1]
                final_tokens_list.append(self.mot_end_idx)
            
            m_tokens_result = np.array(final_tokens_list, dtype=int)
            
            # 5. Calculate m_tokens_len
            m_tokens_len = min(current_length_with_mot_end, self.max_motion_length)

        return caption, m_tokens_result.reshape(-1), m_tokens_len




def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers = 8, sample_way = 0) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, sample_way=sample_way),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    

    return train_loader


def cycle(iterable):
    while True:
        for x in iterable:
            yield x


