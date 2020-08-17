import torch
import torch.utils.data as data
import numpy as np
import pandas as pd
import random
import collections
import json
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def get_rc_data(data, rc_lists):
    new_data = {}
    count = 0
    for k, v in data.items():
        if k in rc_lists:
            new_data[k] = v
            count += len(v)
    logger.info('Remained data numbers: %d' % count)
    return new_data


def RC_align(path):
    def RC_align_out(function):
        def wrap(data_path, threshold, is_train=True):
            rc_list = json.load(open(path, 'r', encoding='utf8'))
            data = function(data_path, threshold, is_train)
            if is_train:
                train_data, eval_data = data
                train_data, eval_data = get_rc_data(train_data, rc_list), get_rc_data(eval_data, rc_list)
                return train_data, eval_data
            else:
                filtered_data = data
                filtered_data = get_rc_data(filtered_data, rc_list)
                return filtered_data
        return wrap
    return RC_align_out


def filter_data(data, threshold):
    filtered_data = {}
    for k, v in data.items():
        if len(v) > threshold:
            filtered_data[k] = v
    logger.info('root cause numbers: %d' % len(filtered_data))
    return filtered_data


@RC_align('./data/rc_list.json')
def read_data(data_path, threshold, is_train=True):
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
    filtered_data = filter_data(data_label, threshold=threshold)
    logger.info("drop %d samples" % count)
    if is_train:
        logger.info("starting samples eval data")
        train_data = collections.defaultdict(list)
        eval_data = collections.defaultdict(list)

        train_data_nums = 0
        eval_data_nums = 0
        for k, v in filtered_data.items():
            eval_nums = len(v) // 10
            if eval_nums == 0:
                eval_nums = 1
            eval_data_nums += eval_nums
            train_data_nums += len(v) - eval_nums
            indices = np.random.choice(list(range(len(v))), eval_nums)
            for j in range(len(v)):
                if j in indices:
                    eval_data[k].append(v[j])
                else:
                    train_data[k].append(v[j])
        logger.info('train data nums: %d, eval data nums: %d' % (train_data_nums, eval_data_nums))
        return train_data, eval_data
    else:
        return filtered_data


class ThinkpadDataset(data.Dataset):
    """
    thinkpad 数据集
    """
    def __init__(self, data, tokenizer, max_seq_len, N, K, Q):
        self.data = data
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
    return_list += [np.zeros_like(inst_data).astype("int64")]

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


def get_loader(train_data, encoder, max_seq_len, N, K, Q, batch_size, num_workers=8, collate_fn=collate_fn):
    dataset = ThinkpadDataset(train_data, encoder, max_seq_len, N, K, Q)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  pin_memory=True,
                                #   num_workers=num_workers,
                                  collate_fn=collate_fn)
    return iter(data_loader)


def output_data_to_excel(data, output_path):
    rc = []
    abstract = []
    des = []
    index = []
    for k, v in data.items():
        for _ in v:
            rc.append(k)
            abstract.append(_[0])
            des.append(_[1])
            index.append(_[2])
    pd_data = pd.DataFrame(data={'abstract': abstract, 'des': des, 'root_cause': rc})
    print(index)
    pd_data.to_excel(output_path)


if __name__ == "__main__":
    np.random.seed(100)
    train_data, eval_data = read_data('./data/source_add_CN_V2.xlsx', 5)
    print(len(train_data.keys()))
    train_data, eval_data = read_data('./data/source_data.xlsx', 5)
    print(len(train_data.keys()))


    # output_data_to_excel(train_data, './data/train.xlsx')
    # output_data_to_excel(eval_data, './data/eval.xlsx')
    # logger.info("heihei")

    # train_rc_emb = json.load(open('./data/train_rc_emb.json', 'r', encoding='utf8'))
    # rc_list = list(train_rc_emb.keys())
    # json.dump(rc_list, open('./data/rc_list.json', 'w', encoding='utf8'), ensure_ascii=False)
