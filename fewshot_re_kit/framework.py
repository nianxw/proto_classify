import os
import sys
import time
from fewshot_re_kit import util
import torch
from torch import nn

# from pytorch_pretrained_bert import BertAdam
from transformers import AdamW, get_linear_schedule_with_warmup
import logging

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def warmup_linear(global_step, warmup_step):
    if global_step < warmup_step:
        return global_step / warmup_step
    else:
        return 1.0


class FewShotREModel(nn.Module):
    def __init__(self, sentence_encoder):
        '''
        sentence_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = nn.DataParallel(sentence_encoder)
        self.cost = nn.CrossEntropyLoss()

    def forward(self, support, query, N, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits, pred
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size. 
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))


class FewShotREFramework:

    def __init__(self,
                 tokenizer=None,
                 train_data_loader=None,
                 val_data_loader=None,
                 test_data_loader=None,
                 train_data=None,
                 eval_data=None):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

        self.train_data = train_data
        self.eval_data = eval_data

        self.tokenizer = tokenizer

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            logger.info("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        torch_version = torch.__version__.split('.')
        if int(torch_version[0]) == 0 and int(torch_version[1]) < 4:
            return x[0]
        else:
            return x.item()

    def eval(self, model, opt):
        train_data_emb, train_rc_emb = util.get_emb(model, self.tokenizer, self.train_data, opt)
        eval_data_emb, _ = util.get_emb(model, self.tokenizer, self.eval_data, opt)
        eval_acc = util.acc(train_data_emb, eval_data_emb, train_rc_emb)
        return eval_acc

    def train(self, model, B, N_for_train, K, Q, opt):
        logger.info("Start training...")

        # Init
        logger.info('Use bert optim!')
        parameters_to_optimize = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        parameters_to_optimize = [
            {'params': [p for n, p in parameters_to_optimize 
                if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in parameters_to_optimize
                if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
            ]
        optimizer = AdamW(parameters_to_optimize, lr=opt.lr, correct_bias=False)

        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=opt.warmup_rate*opt.train_iter, num_training_steps=opt.train_iter)

        start_iter = 0

        model.train()

        # Training
        iter_loss = 0.0
        iter_right = 0.0
        iter_sample = 0.0
        iter_time = 0.0
        begin_time = time.time()
        for it in range(start_iter, start_iter + opt.train_iter):
            batch = next(self.train_data_loader)
            if opt.use_cuda:
                batch = tuple(t.cuda() for t in batch)
            logits, pred = model(batch, N_for_train, K, Q * N_for_train)
            loss = model.loss(logits, batch[-1])
            right = model.accuracy(pred, batch[-1])
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            iter_loss += self.item(loss.data)
            iter_right += self.item(right.data)
            iter_sample += 1

            step_time = time.time() - begin_time
            iter_time += step_time
            sys.stdout.write('step: %d | loss: %.6f, accuracy: %.2f, time/step: %.4f' % (it + 1, iter_loss / iter_sample, 100 * iter_right / iter_sample, iter_time / iter_sample) +'\r')
            sys.stdout.flush()

            if iter_sample % opt.eval_step == 0:
                if opt.do_eval:
                    eval_start_time = time.time()
                    model.eval()
                    eval_model = model.sentence_encoder.module
                    eval_acc = self.eval(eval_model, opt)
                    logger.info("eval used time: %.4f —— eval accuracy: [top1: %.4f] [top3: %.4f] [top5: %.4f]" % (time.time() - eval_start_time, eval_acc[0], eval_acc[1], eval_acc[2]))
                    model.train()

            if iter_sample % opt.save_step == 0:
                logger.info("save model into %s steps: %d" % (opt.save_ckpt, iter_sample))
                torch.save(model.state_dict(), os.path.join(opt.save_ckpt, 'model_%d.bin') % iter_sample)
            begin_time = time.time()
        logger.info("\n####################\n")
        logger.info("Finish training")
        torch.save(model.state_dict(), os.path.join(opt.save_ckpt, 'model_final.bin'))