import torch
import numpy as np
import json


def text_to_tensor(text, tokenizer, max_seq_len):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[: max_seq_len - 2]
    tokens = [tokenizer.cls_token] + tokens + [tokenizer.sep_token]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return torch.tensor([token_ids])


def get_emb(model, tokenizer, data, opt):
    id_to_emd = {}
    root_cause_emb = {}
    for root_cause, samples in data.items():
        rc_emb = []
        for i in range(len(samples)):
            abstract, description, index = samples[i]
            abs_id = text_to_tensor(abstract, tokenizer, opt.max_length)
            des_id = text_to_tensor(description, tokenizer, opt.max_length)
            if opt.use_cuda:
                abs_id = abs_id.cuda()
                des_id = des_id.cuda()
            emb = model(abs_id)[1] + model(des_id)[1]
            emb = emb.view(-1).detach().cpu().numpy().tolist()
            rc_emb.append(emb)
            id_to_emd[index] = [abstract, description, emb, root_cause]
        root_cause_emb[root_cause] = np.mean(np.array(rc_emb), axis=0).tolist()
    return id_to_emd, root_cause_emb


def get_series_emb(emb):
    embeddings = []
    label_to_id = {}
    i = 0
    for index, value in emb.items():
        label_to_id[i] = index
        embeddings.append(value[2])
        i += 1
    return embeddings, label_to_id


def get_rc_emb(emb):
    embeddings = []
    label_to_id = {}
    i = 0
    for index, value in emb.items():
        label_to_id[i] = index
        embeddings.append(value)
        i += 1
    return embeddings, label_to_id


def calculate_distance(S, Q):
    S = S.unsqueeze(0)  # [1, N, D]
    Q = Q.unsqueeze(1)  # [Q, 1, D]
    return -(torch.pow(S - Q, 2)).sum(2)  # [Q, N]


def single_acc(id_to_emd_1, id_to_emd_2):
    '''
    id_to_emd_1: 被查询的emb
    id_to_emd_2: 查询emb
    '''
    emb_1, label_to_id_1 = get_series_emb(id_to_emd_1)
    emb_2, label_to_id_2 = get_series_emb(id_to_emd_2)
    emb_1 = torch.tensor(emb_1)  # train emb
    emb_2 = torch.tensor(emb_2)  # eval emb
    similarity = calculate_distance(emb_1, emb_2)  # [Q, N]

    acc = []
    for k in [1, 3, 5, 10, 50]:
        true_num = 0
        _, indices = similarity.topk(k, dim=-1)
        indices = indices.numpy().tolist()
        for j in range(len(indices)):
            cur_res = indices[j]
            for m in cur_res:
                rc = id_to_emd_1[label_to_id_1[m]][-1]
                if rc == id_to_emd_2[label_to_id_2[j]][-1]:
                    true_num += 1
                    break
        total_num = len(indices)
        acc.append(true_num / total_num)
    return acc


def proto_acc(proto_emb, id_to_emd):
    '''
    id_to_emd: 查询emb
    proto_emb: 原型emb
    '''
    emb, label_to_id = get_series_emb(id_to_emd)
    emb_pro, label_to_rc = get_rc_emb(proto_emb)

    emb_1 = torch.tensor(emb_pro)  # proto emb
    emb_2 = torch.tensor(emb)  # eval emb
    similarity = calculate_distance(emb_1, emb_2)  # [Q, N]

    acc = []
    for k in [1, 3, 5, 10]:
        true_num = 0
        _, indices = similarity.topk(k, dim=-1)
        indices = indices.numpy().tolist()
        for j in range(len(indices)):
            cur_res = indices[j]
            for m in cur_res:
                rc = label_to_rc[m]
                if rc == id_to_emd[label_to_id[j]][-1]:
                    true_num += 1
                    break
        total_num = len(indices)
        acc.append(true_num / total_num)
    return acc


def get_topK_RC(data, K):
    res = []
    for k, v in data.items():
        res.append((k, v))
    res.sort(key=lambda tt: tt[1], reverse=True)
    r = [_[0] for _ in res[: K]]
    return r


def policy_acc(train_data_emb, eval_data_emb, recall_num=100):  # 57
    emb_1, label_to_id_1 = get_series_emb(train_data_emb)
    emb_2, label_to_id_2 = get_series_emb(eval_data_emb)
    emb_1 = torch.tensor(emb_1)  # train emb
    emb_2 = torch.tensor(emb_2)  # eval emb
    similarity = calculate_distance(emb_1, emb_2)  # [Q, N]
    # 对候选结果进行归类，参数设置为30、15、10（若某个类别数据量过少，则会受到候选的影响）

    acc = []
    for x in [1, 3, 5, 10, 50]:
        true_num = 0
        _, indices = similarity.topk(similarity.shape[-1], dim=-1)

        indices = indices.numpy().tolist()  # [Q, N]
        for j in range(len(indices)):
            cur_res = indices[j][:recall_num]
            root_cause_score = {}
            for m in cur_res:
                similar_data = train_data_emb[label_to_id_1[m]]
                rc = similar_data[-1]
                distance = similarity[j][m].item()
                distance = 1 / (-distance + 1e-5)
                if rc not in root_cause_score:
                    root_cause_score[rc] = distance
                else:
                    root_cause_score[rc] += distance
            predict_cause = get_topK_RC(root_cause_score, x)
            cur_case = eval_data_emb[label_to_id_2[j]]
            cur_cause = cur_case[-1]
            if cur_cause in predict_cause:
                true_num += 1
        total_num = len(indices)
        acc.append(true_num / total_num)
    return acc


if __name__ == "__main__":
    train_data_emb = json.load(open('./data/train_emb.json', 'r', encoding='utf8'))
    eval_data_emb = json.load(open('./data/eval_emb.json', 'r', encoding='utf8'))
    train_rc_emb = json.load(open('./data/train_rc_emb.json', 'r', encoding='utf8'))

    acc1 = single_acc(train_data_emb, eval_data_emb)
    print(acc1)
    acc2 = proto_acc(train_rc_emb, eval_data_emb)
    print(acc2)
    acc3 = policy_acc(train_data_emb, eval_data_emb, 57)
    print(acc3)
