import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm



class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, feat_bias = 5, window_size = 64, unit_length = 8):
        self.window_size = window_size
        self.unit_length = unit_length
        self.feat_bias = feat_bias

        self.dataset_name = dataset_name
        self.min_motion_len = 40 if dataset_name =='t2m' else 24
        
        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            radius = 4
            fps = 20
            self.max_motion_length = 196
            dim_pose = 263
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.t2m_kinematic_chain
        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21
            radius = 240 * 8
            fps = 12.5
            dim_pose = 251
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            #kinematic_chain = paramUtil.kit_kinematic_chain
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))
        
        split_file = pjoin(self.data_root, 'train.txt')
        
        self.data_dict = {}
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        self.name_list = []
        self.length_list = []
        self.load_data(id_list)


        self.mean = mean
        self.std = std
        self.length_arr = np.array(self.length_list)
        # self.data_dict = data_dict
        # self.name_list = new_name_list
    
    def load_data(self, id_list):
        import threading
        # self._lock = threading.Lock()
        for name in tqdm(id_list):
            try:
                # with self._lock:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if (len(motion)) < self.min_motion_len or (len(motion) >= 200):
                    continue

                self.name_list.append(name)
                self.length_list.append(len(motion))
                text_data = []
                text_path = pjoin(self.text_dir, name + '.txt')
                with cs.open(text_path, 'r') as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        if len(line_split) > 0:
                            caption = line_split[0]
                            text_data.append(caption)
                self.data_dict[name] = {'motion': motion,
                                'length': len(motion),
                                'name': name,
                                'text': text_data}
            except:
                # Some motion may not exist in KIT dataset
                pass

    def inv_transform(self, data):
        return data * self.std + self.mean

    def __len__(self):
        return len(self.data_dict)

    def __getitem__(self, item):
        name = self.name_list[item]
        data = self.data_dict[name]
        motion, m_length = data['motion'], data['length']
        text = data['text']
        if self.unit_length == 1 and m_length > self.max_motion_length:
            motion = motion[:self.max_motion_length]
            m_length = self.max_motion_length
        else:
            m_length = (m_length // self.unit_length) * self.unit_length
            # if m_length > self.max_motion_length:

            idx = random.randint(0, len(motion) - m_length)
            motion = motion[idx:idx+m_length]
        

        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, name, text

def DATALoader(dataset_name,
                batch_size = 1,
                num_workers = 8, unit_length = 4) : 
    
    train_loader = torch.utils.data.DataLoader(VQMotionDataset(dataset_name, unit_length=unit_length),
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
