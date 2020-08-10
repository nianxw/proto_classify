import torch
import numpy as np


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


def calculate_distance(S, Q):
    S = S.unsqueeze(0)  # [1, N, D]
    Q = Q.unsqueeze(1)  # [Q, 1, D]
    return -(torch.pow(S - Q, 2)).sum(2)  # [Q, N]


def acc(id_to_emd_1, id_to_emd_2, proto_emb=None):
    '''
    emb_1: 被查询的emb
    emb_2: 查询emb
    proto_emb: 原型emb
    '''
    emb_1, label_to_id_1 = get_series_emb(id_to_emd_1)
    emb_2, label_to_id_2 = get_series_emb(id_to_emd_2)
    if proto_emb is not None:
        # proto_emb, label_to_id_2 = get_series_emb(proto_emb)
        pass
    emb_1 = torch.tensor(emb_1)  # train emb
    emb_2 = torch.tensor(emb_2)  # eval emb
    similarity = calculate_distance(emb_1, emb_2)  # [Q, N]

    acc = []
    for k in [1, 3, 5]:
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
