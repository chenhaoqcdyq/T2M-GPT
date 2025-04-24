import json
import os
from typing import List, Dict
import torch
from torch.utils import data
import numpy as np
from os.path import join as pjoin
import random
import codecs as cs
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import BertTokenizer, BertModel
from transformers import CLIPModel, CLIPTokenizer
import re
from functools import partial

class VQMotionDataset(data.Dataset):
    def __init__(self, dataset_name, window_size = 64, unit_length = 4, strategy='basic'):
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
            self.text_mask_dir = pjoin(self.data_root, 'texts_mask_deepseek')

        elif dataset_name == 'kit':
            self.data_root = './dataset/KIT-ML'
            self.motion_dir = pjoin(self.data_root, 'new_joint_vecs')
            self.text_dir = pjoin(self.data_root, 'texts')
            self.joints_num = 21

            self.max_motion_length = 196
            self.meta_dir = 'checkpoints/kit/VQVAEV3_CB1024_CMT_H1024_NRES3/meta'
            self.text_mask_dir = pjoin(self.data_root, 'texts_mask_deepseek')
        
        joints_num = self.joints_num

        mean = np.load(pjoin(self.meta_dir, 'mean.npy'))
        std = np.load(pjoin(self.meta_dir, 'std.npy'))

        split_file = pjoin(self.data_root, 'train.txt')

        self.data = []
        self.lengths = []
        id_list = []
        with cs.open(split_file, 'r') as f:
            for line in f.readlines():
                id_list.append(line.strip())
        name_list = []
        for name in tqdm(id_list):
            try:
                motion = np.load(pjoin(self.motion_dir, name + '.npy'))
                if motion.shape[0] < self.window_size:
                    continue
                self.lengths.append(motion.shape[0] - self.window_size)
                # self.data.append(motion)
                text_data = []
                text_path = pjoin(self.text_dir, name + '.txt')
                # text_tokens_path = pjoin(self.text_token_dir, name + '.pth')
                # if os.path.exists(text_path):
                with cs.open(text_path, 'r') as f:
                    for line in f.readlines():
                        line_split = line.strip().split('#')
                        if len(line_split) > 0:
                            caption = line_split[0]
                            text_data.append(caption)
                self.data.append({
                        'motion': motion,
                        'text': text_data,
                        'name': name
                    })
                name_list.append(name)
            except:
                # Some motion may not exist in KIT dataset
                print("Error: Motion {} not found".format(name))
                pass

        self.mean = mean
        self.std = std
        print("Total number of motions {}".format(len(self.data)))
        self.tokenizer_name = "bert-base-uncased"
        # self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert_model = BertModel.from_pretrained('bert-base-uncased')
        self.bert_model.cuda()
        self.bert_model.eval()  # 冻结BERT参数
        mask_list = [os.path.join(self.text_mask_dir, name + '.json') for name in name_list]
        self.raw_samples = self._load_raw_data(mask_list)
        self.masker = DynamicMaskGenerator(self.bert_tokenizer)
        self.strategy = strategy
        self.static_labels, self.bert_feature = [], []
        # self.text_mask_dir =  pjoin(self.data_root, 'texts_mask_deepseek')
        self.bert_feature_dir = pjoin(self.data_root, 'texts_bert_feature')
        # 生成动态标签
        for i in tqdm(range(len(self.raw_samples))):
            tmp = []
            for j in range(len(self.raw_samples[i])):
                tmp.append(self._get_static_labels(self.raw_samples[i][j]))
            # text_feature_tmp.append()
            self.static_labels.append(tmp)
            self.bert_feature.append(self._get_text_features(self.raw_samples[i], self.bert_feature_dir))
        del self.bert_model

    def inv_transform(self, data):
        return data * self.std + self.mean
    
    def _build_labels(self, encoded: Dict, mask_labels: List, masked_length: List) -> torch.Tensor:
        labels = torch.full_like(encoded['input_ids'], -100)
        tokenized = self.bert_tokenizer.convert_ids_to_tokens(encoded['input_ids'][0])
        
        # 生成有效位置映射表（仅首子词）
        position_map = []
        current_word_pos = 0
        for pos, token in enumerate(tokenized):
            if token in ["[CLS]", "[SEP]"]:
                continue
            # 如果是首子词或者是mask token，添加到映射表
            if not token.startswith("##") and token != self.bert_tokenizer.mask_token:
                position_map.append(pos)
                current_word_pos += 1
            # 如果是mask token且长度大于1，说明是分词后的mask
            elif token == self.bert_tokenizer.mask_token and current_word_pos < len(masked_length) and masked_length[current_word_pos] >= 1:
                position_map.append(pos)
                current_word_pos += 1
        
        # 动态映射原始位置到分词后的首子词位置
        for label in mask_labels:
            original_pos = label['position']  # 原始文本中的位置（从0开始）
            if original_pos >= len(position_map):
                continue
                
            bert_pos = position_map[original_pos]  # 分词后的实际位置
            token_id = encoded['input_ids'][0][bert_pos].item() # 获取实际token ID
        
            if token_id != self.bert_tokenizer.mask_token_id:  # 检查是否为103
                print("Error: Mask token ID not found. bert pose = ", bert_pos," tokenized = ", tokenized, original_pos)
                continue  # 跳过未被MASK的位置
            # 检查是否为子词的首部分
            if tokenized[bert_pos].startswith("##"):
                continue  # 理论上此处不会触发，因position_map已过滤
            length = len(self.bert_tokenizer.tokenize(label['word']))
            labels[0][bert_pos:bert_pos+length] = torch.Tensor(self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(label['word']))).long()
        
        return labels.squeeze()
    
    def _get_text_features(self, sample: Dict, save_dir: str) -> torch.Tensor:
        """获取文本特征"""
        name = sample[0]['name']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
        if not os.path.exists(pjoin(save_dir, name + '.pth')):
            text = []
            for i in range(len(sample)):
                text.append(sample[i]['original_text'])
            encoded = self.bert_tokenizer(
                text,
                padding='max_length',
                truncation=True,
                max_length=128,
                return_tensors='pt'
            )
            for k,v in encoded.items():
                encoded[k] = v.cuda()
            with torch.no_grad():
                output = self.bert_model(**encoded)
            torch.save(output.pooler_output.cpu(), pjoin(save_dir, name + '.pth'))
            torch.save(output.last_hidden_state.cpu(), pjoin(save_dir, name + '_all.pth'))
            result = [output.pooler_output.cpu(), output.last_hidden_state.cpu(), name]
        else:
            result = [torch.load(pjoin(save_dir, name + '.pth'), map_location=torch.device('cpu')), torch.load(pjoin(save_dir, name + '_all.pth'), map_location=torch.device('cpu')), name]
        return result
    
    def _get_static_labels(self, sample: Dict) -> List[Dict]:
        
        """支持重复词位置记录的静态标签生成"""
        labels = []
        # text = sample['original_text'].split()
        text = re.findall(r"\w+|[^\w\s]", sample['original_text'])
        
        # 定义增强版位置查找（处理子词近似匹配）
        def find_all_positions(word: str, tokens: List[str]) -> List[int]:
            positions = []
            for i, token in enumerate(tokens):
                # 处理带标点的单词（如"jumping!" → "jumping"和"!"）
                clean_token = re.sub(r'[^\w]', '', token)
                if clean_token == word:
                    positions.append(i)
            return positions
        
        for action in sample['masked_word']:
            if not action:  # 空动作过滤
                continue
                
            # 处理核心动作词
            core_part = action[0][0].split() if (len(action[0]) > 0 and action[0][0]) else []
            for word in core_part:
                for pos in find_all_positions(word, text):
                    labels.append({
                        'type': 'core',
                        'word': word,
                        'position': pos  # 记录所有位置
                    })
            
            # 处理身体部位词
            if len(action[0]) > 1:
                body_part = action[0][1].split() if action[0][1] else []
                for word in body_part:
                    for pos in find_all_positions(word, text):
                        labels.append({
                            'type': 'body',
                            'word': word,
                            'position': pos
                        })
            
            # 处理方向修饰词
            if len(action) > 1 and action[1]:
                for dir_expr in action[1]:
                    dir_words = dir_expr.split() if dir_expr else []
                    for word in dir_words:
                        for pos in find_all_positions(word, text):
                            labels.append({
                                'type': 'dir',
                                'word': word,
                                'position': pos
                            })
        return labels
    
    def _load_raw_data(self, file_paths: List[str]) -> List[Dict]:
        """加载原始未掩码数据"""
        samples = []
        
        for path in file_paths:
            with open(path, 'r') as f:
                data = json.load(f)
                for i in range(len(data['samples'])):
                    data['samples'][i]['name'] = os.path.basename(path).split('.')[0] 
                samples.append(data['samples'])
        return samples
    
    def compute_sampling_prob(self) : 
        
        prob = np.array(self.lengths, dtype=np.float32)
        prob /= np.sum(prob)
        return prob
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        data = self.data[item]
        motion = data['motion']
        text_list = data['text']
        name = data['name']
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
        
        text_random_id = random.randint(0, len(text_list) - 1)
        text = text_list[text_random_id]
        raw_sample_list = self.raw_samples[item]
        idx_sample = text_random_id
        if len(text_list) != len(raw_sample_list):
            print('text_list:', text_list)
        raw_sample = raw_sample_list[idx_sample]
        text = raw_sample['original_text']
        static_labels = self.static_labels[item][idx_sample]
        text_bert_feature = self.bert_feature[item][0][idx_sample]
        text_bert_feature_all = self.bert_feature[item][1][idx_sample]
        # 选择掩码策略
        if self.strategy == "progressive":
            stage = random.choice(self.masker.progressive_stages)
        else:
            stage = self.strategy
        # if self.args.lgvq>0:
        # 动态生成掩码
        masked_data = self.masker.generate_masks(text, static_labels, stage)
        
        # 编码处理
        encoded = self.bert_tokenizer(
            masked_data['masked_text'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        # 构建动态标签
        labels = self._build_labels(encoded, masked_data['labels'], masked_data['masked_length'])
        text_mask = {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': labels,
            'feature': text_bert_feature,
            'feature_all': text_bert_feature_all
        }

        return motion, motion_mask, text_mask, name

class DynamicMaskGenerator:
    def __init__(self, 
                 bert_tokenizer,
                 progressive_stages: List[str] = ['basic', 'medium', 'advanced']):
        self.strategies = {
            'basic': partial(self._mask_core_dir, core_prob=0.5, dir_prob=0.3),
            'medium': partial(self._mask_core_body_dir, core_prob=0.6, dir_prob=0.4, body_prob=0.2),
            'advanced': self._full_mask,
            'random': self._random_mask
        }
        self.bert_tokenizer = bert_tokenizer
        self.mask_token = "[MASK]"
        self.progressive_stages = progressive_stages
        
    def generate_masks(self, text: str, labels: List, strategy: str = 'random') -> Dict:
        """动态生成掩码"""
        words = re.findall(r"\w+|[^\w\s]", text)
        masked = words.copy()
        new_labels = []
        
        if strategy in self.strategies:
            masked, new_labels = self.strategies[strategy](words, labels)
        else:
            masked, new_labels = self._random_mask(words, labels)
        masked_text = words.copy()
        for i in range(len(masked)):
            if masked[i] != 0:
                masked_text[i] = (self.mask_token + " ") * masked[i]
            if masked[i] > 1:
                pass
        return {
            'masked_text': ' '.join(masked_text),
            'labels': new_labels,
            'masked_length': masked
        }

    def _mask_core(self, words, labels, prob=0.3):
        """核心动作掩码"""
        # masked = words.copy()
        masked = torch.zeros(len(words), dtype=torch.int64)
        new_labels = []
        for word_info in labels:
            if word_info['type'] == 'core' and random.random() < prob:
                pos = word_info['position']
                try:
                    masked[pos] = len(self.bert_tokenizer.tokenize(word_info['word']))
                except:
                    print('pos:', pos)
                    print('len(masked):', len(masked))
                    print('masked:', masked)
                new_labels.append(word_info)
        return masked, new_labels

    def _mask_core_dir(self, words, labels, core_prob=0.5, dir_prob=0.3):
        """核心+方向联合掩码"""
        masked, new_labels = self._mask_core(words, labels, core_prob)
        for word_info in labels:
            if word_info['type'] == 'dir' and random.random() < dir_prob:
                pos = word_info['position']
                try:
                    # masked[pos] = self.mask_token
                    masked[pos] = len(self.bert_tokenizer.tokenize(word_info['word']))
                except:
                    print('pos:', pos)
                    print('len(masked):', len(masked))
                    print('masked:', masked)
                new_labels.append(word_info)
        return masked, new_labels

    def _mask_core_body_dir(self, words, labels, core_prob=0.5, dir_prob=0.3, body_prob=0.3):
        """核心+方向联合掩码"""
        masked, new_labels = self._mask_core_dir(words, labels, core_prob, dir_prob)
        for word_info in labels:
            if word_info['type'] == 'body' and random.random() < body_prob:
                pos = word_info['position']
                # masked[pos] = self.mask_token
                masked[pos] = len(self.bert_tokenizer.tokenize(word_info['word']))
                new_labels.append(word_info)
        return masked, new_labels

    def _full_mask(self, words, labels):
        """全要素渐进式掩码"""
        return self._mask_core_body_dir(words, labels, 0.8, 0.7, 0.5)

    def _random_mask(self, words, labels):
        """随机掩码策略"""
        masked = words.copy()
        new_labels = []
        for word_info in labels:
            if random.random() < 0.3:  # 基础概率
                pos = word_info['position']
                # masked[pos] = self.mask_token
                masked[pos] = len(self.bert_tokenizer.tokenize(word_info['word']))
                new_labels.append(word_info)
        return masked, new_labels

def DATALoader(dataset_name,
               batch_size,
               num_workers = 8,
               window_size = 64,
               unit_length = 4):
    
    trainSet = VQMotionDataset(dataset_name, window_size=window_size, unit_length=unit_length)
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
