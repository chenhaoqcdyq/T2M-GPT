import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
import utils.paramUtil as paramUtil
from torch.utils.data._utils.collate import default_collate
import os

def collate_fn(batch):
    batch.sort(key=lambda x: x[3], reverse=True)
    return default_collate(batch)


'''For use of training text-2-motion generative model'''
class Text2MotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, unit_length = 4, codebook_size = 1024, tokenizer_name=None, test_nb=False):
        
        self.max_length = 64
        self.pointer = 0
        self.dataset_name = dataset_name

        self.unit_length = unit_length
        # self.mot_start_idx = codebook_size
        self.codebook_size = codebook_size
        self.mot_end_idx = codebook_size
        self.mot_pad_idx = codebook_size + 1
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            self.fps = 20
            self.max_motion_length = 196 // unit_length + 2
            dim_pose = 263
            kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            self.fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196 // unit_length + 2
            kinematic_chain = paramUtil.kit_kinematic_chain

        split_file = pjoin(self.data_root, 'train.txt')

        self.test_nb = test_nb
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        self.tokenizer_name = tokenizer_name
        self.name_list = []
        self.data_dict = {}
        self.load_data(id_list) 
        # self.data_dict = data_dict
        # self.name_list = new_name_list
        
    def load_data(self, id_list):
        # import threading
        # self._lock = threading.Lock()
        for name in tqdm(id_list):
            try:
                # with self._lock:
                if os.path.exists(pjoin(self.data_root, self.tokenizer_name, '%s.npz'%name)):
                    m_token_list = np.load(pjoin(self.data_root, self.tokenizer_name, '%s.npz'%name))
                    m_token_list = m_token_list['motion']
                elif os.path.exists(pjoin(self.data_root, self.tokenizer_name, '%s.npy'%name)):
                    m_token_list = np.load(pjoin(self.data_root, self.tokenizer_name, '%s.npy'%name))
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
                                m_token_list_new = [tokens[int(f_tag*self.fps/self.unit_length) : int(to_tag*self.fps/self.unit_length)] for tokens in m_token_list if int(f_tag*self.fps/self.unit_length) < int(to_tag*self.fps/self.unit_length)]

                                if len(m_token_list_new) == 0:
                                    continue
                                new_name = '%s_%f_%f'%(name, f_tag, to_tag)

                                self.data_dict[new_name] = {'m_token_list': m_token_list_new,
                                                    'text':[text_dict]}
                                self.name_list.append(new_name)
                        except:
                            pass

                if flag:
                    self.data_dict[name] = {'m_token_list': m_token_list,
                                    'text':text_data}
                    self.name_list.append(name)
            except:
                pass
            
    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        data = self.data_dict[self.name_list[item]]
        m_token_list, text_list = data['m_token_list'], data['text']
        m_tokens = random.choice(m_token_list)
        
        text_data = random.choice(text_list)
        caption= text_data['caption']

        if len(m_tokens.shape) == 2:
            m_tokens = m_tokens.transpose(1, 0)
            # m_tokens (6, 196)
            coin = np.random.choice([False, False, True])
            # print(len(m_tokens))
            if coin:
                # drop one token at the head or tail
                coin2 = np.random.choice([True, False])
                if coin2:
                    m_tokens = m_tokens[:, :-1]
                else:
                    m_tokens = m_tokens[:, 1:]
            m_tokens_len = m_tokens.shape[1]

            if m_tokens_len+1 < self.max_motion_length:
                m_tokens = [np.concatenate([m_tokens[i], np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0) for i in range(m_tokens.shape[0])]
            else:
                m_tokens = [np.concatenate([m_tokens[i], np.ones((1), dtype=int) * self.mot_end_idx], axis=0) for i in range(m_tokens.shape[0])]
            m_tokens = np.stack(m_tokens, axis=0)
            return caption, m_tokens, m_tokens_len
        else:
            coin = np.random.choice([False, False, True])
            # print(len(m_tokens))
            if coin:
                # drop one token at the head or tail
                coin2 = np.random.choice([True, False])
                if coin2:
                    m_tokens = m_tokens[:-1]
                else:
                    m_tokens = m_tokens[1:]
            m_tokens_len = m_tokens.shape[0]

            if m_tokens_len+1 < self.max_motion_length:
                m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx, np.ones((self.max_motion_length-1-m_tokens_len), dtype=int) * self.mot_pad_idx], axis=0)
            else:
                m_tokens = np.concatenate([m_tokens, np.ones((1), dtype=int) * self.mot_end_idx], axis=0)
        if self.test_nb:
            m_tokens = m_tokens + self.codebook_size + 1 # 513

        return caption, m_tokens.reshape(-1), m_tokens_len




def DATALoader(dataset_name,
                batch_size, codebook_size, tokenizer_name, unit_length=4,
                num_workers = 8, sample_way = 0, test_nb=False) : 

    train_loader = torch.utils.data.DataLoader(Text2MotionDataset(dataset_name, codebook_size = codebook_size, tokenizer_name = tokenizer_name, unit_length=unit_length, test_nb=test_nb),
                                              batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    

    return train_loader

def DATALoader_ddp(dataset_name, batch_size, codebook_size, tokenizer_name, unit_length=4, num_workers=4, sample_way=0):
    # 返回数据集实例而不是DataLoader
    return Text2MotionDataset(
        dataset_name, 
        codebook_size=codebook_size, 
        tokenizer_name=tokenizer_name, 
        unit_length=unit_length, 
        sample_way=sample_way
    )

def cycle(iterable):
    while True:
        for x in iterable:
            yield x


