import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm



class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4, test=False):
        self.window_size = window_size
        self.unit_length = unit_length
        self.dataset_name = dataset_name

        if dataset_name == 't2m':
            self.data_root = './dataset/HumanML3D'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 22
            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/t2m/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, 'train.txt')
        if test:
            split_file = pjoin(self.data_root, 'test.txt')
        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())

        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                self.data.append(motion)
            except:
                # Some motion may not exist in KIT dataset
                pass

            
        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        motion = self.data[item]
        motion_mask = np.zeros((self.max_motion_length,))
        if len(motion) < self.max_motion_length:
            motion_mask[:len(motion)] = 1
            motion = np.concatenate([motion, np.zeros((self.max_motion_length - len(motion), motion.shape[1]))], axis=0)
        else:
            motion = motion[:self.max_motion_length]
            motion_mask[:self.max_motion_length] = 1
        # idx = random.randint(0, len(motion) - self.window_size)
        
        # motion = motion[idx:idx+self.window_size]
        "Z Normalization"
        motion = (motion - self.mean) / self.std

        return motion, motion_mask

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4,
               test=False):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length, test=test)
    prob = trainSet.compute_sampling_prob()
    sampler = torch.utils.data.WeightedRandomSampler(prob, num_samples = len(trainSet) * 1000, replacement=True)
    train_loader = torch.utils.data.DataLoader(trainSet,
                                              batch_size,
                                              shuffle=True,
                                              #sampler=sampler,
                                              num_workers=num_workers,
                                              #collate_fn=collate_fn,
                                              drop_last = True)
    
    return train_loader

def cycle(iterable):
    while True:
        for x in iterable:
            yield x
