from fewshot_re_kit.data_loader import get_loader, read_data
from fewshot_re_kit.framework import FewShotREFramework
import models
from models.proto import Proto
import sys
import torch
from torch import optim, nn
import numpy as np
import argparse
import os
from transformers import AdamW, BertModel, BertConfig, BertTokenizer
import logging
import test

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--do_train', default=False, type=bool)
    parser.add_argument('--do_predict', default=True, type=bool)
    parser.add_argument('--train_file', default='./data/train.xlsx',
            help='train file')
    parser.add_argument('--trainN', default=5, type=int,
            help='N in train')
    parser.add_argument('--N', default=5, type=int,
            help='N way')
    parser.add_argument('--K', default=3, type=int,
            help='K shot')
    parser.add_argument('--Q', default=3, type=int,
            help='Num of query per class')
    parser.add_argument('--batch_size', default=8, type=int,
            help='batch size')
    parser.add_argument('--train_iter', default=10000, type=int,
            help='num of iters in training')
    parser.add_argument('--warmup_rate', default=0.1, type=float)
    parser.add_argument('--max_length', default=128, type=int,
           help='max length')
    parser.add_argument('--lr', default=2e-5, type=float,
           help='learning rate')
    parser.add_argument('--dropout', default=0.0, type=float,
           help='dropout rate')
    parser.add_argument('--hidden_size', default=230, type=int,
           help='hidden size')
    parser.add_argument('--load_ckpt', default='./check_points/model_10000.bin',
           help='load ckpt')
    parser.add_argument('--save_ckpt', default='./check_points/',
           help='save ckpt')
    parser.add_argument('--save_emb', default='./data/emb.json',
           help='save embedding')
    parser.add_argument('--save_root_emb', default='./data/root_emb.json',
           help='save embedding')
    parser.add_argument('--use_cuda', default=True,
           help='whether to use cuda')
    parser.add_argument('--save_step', default=2000,
           help='whether to use cuda')
    # bert pretrain
    parser.add_argument("--vocab_file", default="./pretrain/vocab.txt", type=str, help="Init vocab to resume training from.")
    parser.add_argument("--config_path", default="./pretrain/bert_config.json", type=str, help="Init config to resume training from.")
    parser.add_argument("--init_checkpoint", default="./pretrain/pytorch_model.bin", type=str, help="Init checkpoint to resume training from.")

    opt = parser.parse_args()
    trainN = opt.trainN
    K = opt.K
    Q = opt.Q
    batch_size = opt.batch_size
    max_length = opt.max_length

    logger.info("{}-way-{}-shot Few-Shot Dignose".format(trainN, K))
    logger.info("max_length: {}".format(max_length))

    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_config = BertConfig.from_pretrained('bert-base-cased')
    bert_model = BertModel.from_pretrained(opt.init_checkpoint, config=bert_config)
    model = Proto(bert_model, opt)
    if opt.use_cuda:
        model.cuda()

    if opt.do_train:
        train_data_loader = get_loader(opt.train_file, bert_tokenizer, max_length, N=trainN, K=K, Q=Q, batch_size=batch_size)

        framework = FewShotREFramework(train_data_loader=train_data_loader)
        framework.train(model, batch_size, trainN, K, Q, opt)

    if opt.do_predict:
        id_to_emd, root_cause_emb = test.predict(opt, bert_model=bert_model, bert_tokenizer=bert_tokenizer)
        test.test_acc(id_to_emd)


if __name__ == "__main__":
    main()
