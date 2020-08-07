import torch
import torch.utils.data as data
import os
import numpy as np
import pandas as pd
import random
import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def filter_data(data, threshold=5):
    filtered_data = {}
    for k, v in data.items():
        if len(v) > 5:
            filtered_data[k] = v
    return filtered_data


def read_data(data_path):
    data = pd.read_excel(data_path, sheet_name='Sheet1', header=[0], usecols='A,B,C').fillna(0)
    data_label = {}
    count = 0
    for i in range(len(data)):
        if data.iloc[i]["Abstract"] == 0 or data.iloc[i]["Problem Description"] == 0:
            count += 1
            continue
        text = [data.iloc[i]["Abstract"], data.iloc[i]["Problem Description"], i]
        label = data.iloc[i]["Root Cause"]
        if label not in data_label:
            data_label[label] = [text]
        else:
            data_label[label].append(text)
    filtered_data = filter_data(data_label, threshold=5)
    logger.info("drop %d samples" % count)
    return filtered_data


class ThinkpadDataset(data.Dataset):
    """
    thinkpad 数据集
    """
    def __init__(self, file_path, tokenizer, max_seq_len, N, K, Q):
        self.data = read_data(file_path)
        self.classes = list(self.data.keys())
        self.tokenizer = tokenizer
        self.cls_token = self.tokenizer.cls_token
        self.sep_token = self.tokenizer.sep_token
        self.max_seq_len = max_seq_len
        self.N = N
        self.K = K
        self.Q = Q

    def tokenize(self, text):
        tokens = self.tokenizer.tokenize(text)

        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[: self.max_seq_len - 2]

        tokens = [self.cls_token] + tokens + [self.sep_token]
        token_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        return token_ids

    def __getitem__(self, idx):
        target_classes = random.sample(self.classes, self.N)
        support_abstract_set = []
        support_description_set = []
        query_abstract_set = []
        query_description_set = []
        query_label = []
        for i, class_name in enumerate(target_classes):
            indices = np.random.choice(
                    list(range(len(self.data[class_name]))),
                    self.K + self.Q, False)  # 针对每种类别随机抽出K+Q条数据
            count = 0
            for j in indices:
                abstract, description, _ = self.data[class_name][j]
                abstract = self.tokenize(abstract)
                description = self.tokenize(description)
                if count < self.K:
                    support_abstract_set.append(abstract)  # 为support set加入数据
                    support_description_set.append(description)
                else:
                    query_abstract_set.append(abstract)  # 为query set加入数据
                    query_description_set.append(description)
                count += 1
            query_label += [i] * self.Q
        return support_abstract_set, support_description_set, query_abstract_set, query_description_set, query_label

    def __len__(self):
        return 1000000000


def pad_seq(insts):
    return_list = []

    max_len = max(len(inst) for inst in insts)

    # input ids
    inst_data = np.array(
        [inst + list([0] * (max_len - len(inst))) for inst in insts],
    )
    return_list += [inst_data.astype("int64")]

    # input sentence type
    return_list += [np.zeros_like(inst_data)]

    # input position
    inst_pos = np.array([list(range(0, len(inst))) + [0] * (max_len - len(inst)) for inst in insts])
    return_list += [inst_pos.astype("int64")]

    # input mask
    input_mask_data = np.array([[1] * len(inst) + [0] * (max_len - len(inst)) for inst in insts])
    return_list += [input_mask_data.astype("float32")]

    return return_list


def collate_fn(data):
    # [batch_size, N*K, seq_len] \ [batch_size, N*Q, seq_len]
    support_abstract_sets, support_description_sets, query_abstract_sets, query_description_sets, query_labels = zip(*data)
    batch_s_abs = []
    batch_s_des = []
    batch_q_abs = []
    batch_q_des = []
    batch_q_label = []
    for i in range(len(support_abstract_sets)):
        batch_s_abs.extend(support_abstract_sets[i])
        batch_s_des.extend(support_description_sets[i])
        batch_q_abs.extend(query_abstract_sets[i])
        batch_q_des.extend(query_description_sets[i])
        batch_q_label.extend(query_labels[i])
    padded_token_s_abs_ids, padded_text_type_s_abs_ids, padded_position_s_abs_ids, input_s_abs_mask = pad_seq(batch_s_abs)
    padded_token_s_des_ids, padded_text_type_s_des_ids, padded_position_s_des_ids, input_s_des_mask = pad_seq(batch_s_des)
    padded_token_q_abs_ids, padded_text_type_q_abs_ids, padded_position_q_abs_ids, input_q_abs_mask = pad_seq(batch_q_abs)
    padded_token_q_des_ids, padded_text_type_q_des_ids, padded_position_q_des_ids, input_q_des_mask = pad_seq(batch_q_des)

    return_list = [
            padded_token_s_abs_ids, padded_text_type_s_abs_ids, padded_position_s_abs_ids, input_s_abs_mask, 
            padded_token_s_des_ids, padded_text_type_s_des_ids, padded_position_s_des_ids, input_s_des_mask,
            padded_token_q_abs_ids, padded_text_type_q_abs_ids, padded_position_q_abs_ids, input_q_abs_mask,
            padded_token_q_des_ids, padded_text_type_q_des_ids, padded_position_q_des_ids, input_q_des_mask,
            batch_q_label
            ]
    return_list = [torch.tensor(batch_data) for batch_data in return_list]
    return return_list

def get_loader(file_path, encoder, max_seq_len, N, K, Q, batch_size, num_workers=8, collate_fn=collate_fn):
    dataset = ThinkpadDataset(file_path, encoder, max_seq_len, N, K, Q)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                #   num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


if __name__ == "__main__":
    data = read_data('./data/train.xlsx')
    logger.info("dd")