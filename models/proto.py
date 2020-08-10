import torch
import torch.nn as nn
import fewshot_re_kit


class Proto(fewshot_re_kit.framework.FewShotREModel):
    def __init__(self, sentence_encoder, config):
        fewshot_re_kit.framework.FewShotREModel.__init__(self, sentence_encoder)
        self.drop = nn.Dropout(config.dropout)

    def __dist__(self, x, y, dim):
        return -(torch.pow(x - y, 2)).sum(dim)

    def __batch_dist__(self, S, Q):
        return self.__dist__(S.unsqueeze(1), Q.unsqueeze(2), 3)

    def forward(self, batch, N, K, total_Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        K: Num of instances for each class in the support set
        Q: Num of instances in the query set
        '''
        support_abs_emb = self.sentence_encoder(batch[0], batch[3], batch[1], batch[2])[1]  # (B * N * K, D), where D is the hidden size
        support_des_emb = self.sentence_encoder(batch[4], batch[7], batch[5], batch[6])[1]
        support_emb = support_abs_emb + support_des_emb

        query_abs_emb = self.sentence_encoder(batch[8], batch[11], batch[9], batch[10])[1]
        query_des_emb = self.sentence_encoder(batch[12], batch[15], batch[13], batch[14])[1]
        query_emb = query_abs_emb + query_des_emb  # (B * total_Q, D)

        hidden_size = support_emb.size(-1)
        support = self.drop(support_emb)
        query = self.drop(query_emb)
        support = support.view(-1, N, K, hidden_size)  # (B, N, K, D)
        query = query.view(-1, total_Q, hidden_size)  # (B, total_Q, D)

        B = support.size(0)  # Batch size

        # Prototypical Networks
        support = torch.mean(support, 2)  # Calculate prototype for each class
        logits = self.__batch_dist__(support, query)  # (B, total_Q, N)
        _, pred = torch.max(logits.view(-1, N), 1)
        return logits, pred