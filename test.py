from fewshot_re_kit.data_loader import read_data
import torch
import json
import os
from tqdm import tqdm
import numpy as np
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def text_to_tensor(text, bert_tokenizer, max_seq_len):
    tokens = bert_tokenizer.tokenize(text)
    if len(tokens) > max_seq_len - 2:
        tokens = tokens[: max_seq_len - 2]
    tokens = [bert_tokenizer.cls_token] + tokens + [bert_tokenizer.sep_token]
    token_ids = bert_tokenizer.convert_tokens_to_ids(tokens)
    return torch.tensor([token_ids])


def predict(opt, bert_model, bert_tokenizer):
    if not os.path.exists(opt.save_emb):
        state_dict = torch.load(opt.load_ckpt)
        own_state = bert_model.state_dict()
        for name, param in state_dict.items():
            name = name.replace('sentence_encoder.module.', '')
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        bert_model.eval()
        root_dict = read_data(opt.train_file)
        id_to_emd = {}
        root_cause_emb = {}
        for root_cause, samples in tqdm(root_dict.items()):
            rc_emb = []
            for i in range(len(samples)):
                abstract, description, index = samples[i]
                abs_id = text_to_tensor(abstract, bert_tokenizer, opt.max_length)
                des_id = text_to_tensor(description, bert_tokenizer, opt.max_length)
                if opt.use_cuda:
                    abs_id = abs_id.cuda()
                    des_id = des_id.cuda()
                emb = bert_model(abs_id)[1] + bert_model(des_id)[1]
                emb = emb.view(-1).detach().cpu().numpy().tolist()
                rc_emb.append(emb)
                id_to_emd[index] = [abstract, description, emb, root_cause]
            root_cause_emb[root_cause] = np.mean(np.array(rc_emb), axis=0).tolist()
        json.dump(id_to_emd, open(opt.save_emb, 'w', encoding='utf8'), ensure_ascii=False)
        json.dump(root_cause_emb, open(opt.save_root_emb, 'w', encoding='utf8'), ensure_ascii=False)
    else:
        id_to_emd = json.load(open(opt.save_emb, 'r', encoding='utf8'))
        root_cause_emb = json.load(open(opt.save_emb, 'r', encoding='utf8'))
    return id_to_emd, root_cause_emb


def calculate_distance(S, Q):
    S = S.unsqueeze(0)  # [1, N, D]
    Q = Q.unsqueeze(1)  # [Q, 1, D]
    return -(torch.pow(S - Q, 2)).sum(2)


def test_acc(id_to_emd=None):
    if id_to_emd is None:
        id_to_emd = json.load(open('./data/emb.json', 'r', encoding='utf8'))
    embeddings = []
    label_to_id = {}
    i = 0
    for index, value in id_to_emd.items():
        label_to_id[i] = index
        embeddings.append(value[2])
        i += 1
    embeddings = torch.tensor(embeddings)  # [N, 768]
    similarity = calculate_distance(embeddings, embeddings)  # [N, N]
    mask = torch.triu(torch.ones_like(similarity), diagonal=1)
    mask = mask + mask.transpose(1, 0)
    mask = (1 - mask) * -10000
    similarity = similarity + mask

    for k in [1, 3, 5]:
        true_num = 0
        _, indices = similarity.topk(k, dim=-1)
        indices = indices.numpy().tolist()
        for j in range(len(indices)):
            cur_res = indices[j]
            for m in cur_res:
                rc = id_to_emd[label_to_id[m]][-1]
                if rc == id_to_emd[label_to_id[j]][-1]:
                    true_num += 1
                    break
        total_num = len(indices)
        acc = true_num / total_num
        logging.info('top %d : accuracy —— %.4f' % (k, acc))


def classification_test(id_to_emd=None, root_cause_emb=None):
    if id_to_emd is None and root_cause_emb is None:
        id_to_emd = json.load(open('./data/emb.json', 'r', encoding='utf8'))
        root_cause_emb = json.load(open('./data/root_emb.json', 'r', encoding='utf8'))
    pass


if __name__ == "__main__":
    test_acc()
    print('OK')