import math


from approx_ndcg_loss import approx_ndcg_loss

from LightGCN import LightGCN
from CourseDataset import *
import torch.nn as nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import torch.nn.functional as F
from Parse import parse_mooc_args
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from BarlowTwinsLoss import BarlowTwinsLoss


class FAME(nn.Module):
    def __init__(self, args, Graph1, Graph2):
        super(FAME, self).__init__()
        self.num_student = args.num_student
        self.num_category = args.num_category
        self.num_course = args.num_course
        self.device = args.device
        self.sequence_length = args.sequence_length
        self.keep_prob = args.keep_prob
        self.latent_dim = args.recdim
        self.dropout = args.dropout
        self.transformer = args.transformer
        self.nhead = args.nhead
        self.cosine_loss = nn.CosineEmbeddingLoss()
        # 初始化两个LightGCN实例
        self.lightgcn1 = LightGCN(
            args.num_student, args.num_course, args.recdim,
            args.layer, args.keep_prob, args.dropout, Graph1, args.device
        )
        self.lightgcn2 = LightGCN(
            args.num_student, args.num_course, args.recdim,
            args.layer, args.keep_prob, args.dropout, Graph2, args.device
        )
        # 其他组件初始化
        self.category_embedding = nn.Embedding(self.num_category + 1, self.latent_dim, padding_idx=0)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.cl = BarlowTwinsLoss()

        self.warmup_epochs=15
        self.tau_start = 2.0
        self.tau_end = 0.5



        if self.transformer:
            self.embeddings_position = nn.Embedding(self.sequence_length + 1, self.latent_dim)
            self.transfomerlayer = TransformerLayerMoH(
                d_model=self.latent_dim, d_ff=512,
                n_heads=self.nhead,

                dropout=0.2
            )
            self.liner = nn.Sequential(
                nn.Linear(22*self.latent_dim, 1024),#2200,输入22*embedding大小
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(1024, 512),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(512, 256),
                nn.LeakyReLU(),
                nn.Dropout(p=self.dropout),
                nn.Linear(256, 1),
            )

        self._init_weight_()
        self.to(self.device)

    def _init_weight_(self):
        nn.init.normal_(self.category_embedding.weight, std=0.001)
        if self.transformer:
            nn.init.normal_(self.embeddings_position.weight, std=0.001)
            for m in self.liner:
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
            for m in self.modules():
                if isinstance(m, nn.Linear) and m.bias is not None:
                    m.bias.data.zero_()

    def getEmbedding_test(self, student, courses, candidate):
        # 1. GCN 输出
        users1, items1 = self.lightgcn1.computer()  # 图一
        users2, items2 = self.lightgcn2.computer()  # 图二

        # 对比学习（在对齐后的 item 语义空间中做两个图之间的对比）
        iloss = self.cl(items1, items2)  # 结构对比：图一课程 vs 图二课程

        #对齐后的用户与课程 embedding
        all_users = users1
        #all_items = 0.5 * items1 + 0.5 * items2  # 图一 & 图二课程对齐后的融合
        all_items = 0.5 * items1 + 0.5 * items2
        # 返回推荐模型用的表示和总损失
        return (
            all_users[student],  # 对齐后的 student 表示
            all_items[courses],  # 对齐后的历史课程表示
            all_items[candidate.long()],  # ⬅ 对齐后的候选课程表示
            iloss  # 总损失：对齐 + 图间对比

        )
    def getEmbedding_train(self, student, courses, candidate, neg_candidate):
        # 1. GCN 输出
        users1, items1 = self.lightgcn1.computer()
        users2, items2 = self.lightgcn2.computer()

        # 2. 对比学习损失
        cl_loss = self.cl(items1, items2)

        # 3. 对齐后 embedding
        all_users = users1
        all_items = 0.5 * items1 + 0.5 * items2

        # 4. 正负样本 embedding
        u_emb = all_users[student]                     # [B, D]
        i_emb = all_items[courses]              # [B, D]
        can_emb = all_items[candidate.long()]          # [B, D]
        neg_embs = all_items[neg_candidate.long()]     # [B, N, D], N=4



        return u_emb, i_emb,can_emb,neg_embs, cl_loss

    def forward(self, student, courses, category, candidate, candidate_cate,if_train,neg_candidate=None,neg_category=None,pop_dict=None):
        if if_train:
            # 获取嵌入
            user_emb, hist_emb, cand_emb, neg_embs, cl_loss = self.getEmbedding_train(student, courses, candidate,
                                                                                      neg_candidate)

            # 历史课程 + 类别特征
            category_emb = self.category_embedding(category)  # [B, S, D]
            course_feature = hist_emb + category_emb  # [B, S, D]

            # 正样本特征
            candidate_cate_emb = self.category_embedding(candidate_cate)  # [B, D]
            pos_feature = cand_emb + candidate_cate_emb  # [B, D]

            # 负样本特征（新的 neg_embs）
            neg_cate_embs = self.category_embedding(neg_category)  # [B, N, D]
            neg_features = neg_embs + neg_cate_embs  # [B, N, D]


            all_embs = torch.cat([cand_emb.unsqueeze(1), neg_embs], dim=1)  # [B, N+1, D]
            # 合并正负样本特征
            all_features = torch.cat([pos_feature.unsqueeze(1), neg_features], dim=1)  # [B, N+1, D]
            batch_size, num_candidates, dim = all_features.shape  # num_candidates = N+1
            seq_len = course_feature.size(1)
            total_seq = seq_len + 1  # 历史 + 候选

            #=====冷热门度编码=====
            pop_expand = pop_dict.unsqueeze(-1).repeat(1, 1, total_seq)  # [B, N+1, S+1]
            pop_seq = pop_expand.view(batch_size * num_candidates, total_seq)  # [B*(N+1), S+1]


            # 准备 transformer 输入
            if self.transformer:
                # 位置编码
                positions = self.embeddings_position(torch.arange(total_seq, device=self.device))  # [S+1, D]
                positions = positions.unsqueeze(0).repeat(batch_size * num_candidates, 1, 1)  # [B*(N+1), S+1, D]

                # course_feature 扩展复制给每个候选
                repeated_hist = course_feature.unsqueeze(1).repeat(1, num_candidates, 1, 1)  # [B, N+1, S, D]
                all_features = all_features.unsqueeze(2)  # [B, N+1, 1, D]
                transformer_input = torch.cat([repeated_hist, all_features], dim=2)  # [B, N+1, S+1, D]
                transformer_input = transformer_input.view(batch_size * num_candidates, total_seq, dim)
                transformer_input = transformer_input + positions

                # 输入 Transformer
                seq_out, gating = self.transfomerlayer(transformer_input, transformer_input, transformer_input,pop_seq=pop_seq)
                seq_out = torch.flatten(seq_out, 1)  # [B*(N+1), (S+1)*D]
            else:
                # 非 transformer 情况（少见）
                seq_out = torch.cat([
                    course_feature.unsqueeze(1).repeat(1, num_candidates, 1, 1),
                    all_features.unsqueeze(2)
                ], dim=2).view(batch_size * num_candidates, -1)

            # 拼接 user_emb
            user_rep = user_emb.unsqueeze(1).repeat(1, num_candidates, 1).view(batch_size * num_candidates, -1)



            final_feature = torch.cat([user_rep, seq_out], dim=1)  # [B*(N+1), D']

            scores = self.liner(final_feature).view(batch_size, num_candidates)

            # 计算 listwise loss
            labels = torch.zeros_like(scores)
            labels[:, 0] = 1.0  # 正样本 index=0
            listwise_loss = approx_ndcg_loss(scores, labels, tau=2.0)



            # —— 新增：负载均衡损失 Lb —— #
            # 此部分代码暂未公开

            return student, candidate, scores, cl_loss +10*listwise_loss + 10*Lb
        else:
            # 获取嵌入特征
            user_emb, hist_emb, cand_emb, cl_loss = self.getEmbedding_test(student, courses, candidate)

            # 类别特征处理
            category_emb = self.category_embedding(category)
            course_feature = hist_emb + category_emb

            # 候选特征处理
            candidate_cate_emb = self.category_embedding(candidate_cate)
            candidate_feature = cand_emb + candidate_cate_emb

            if self.transformer:
                # Transformer处理时序特征
                positions = self.embeddings_position(
                    torch.arange(self.sequence_length + 1, device=self.device)
                )
                sequence = torch.cat([course_feature, candidate_feature.unsqueeze(1)], dim=1)
                sequence = sequence + positions

                total_seq = self.sequence_length + 1
                pop_seq = pop_dict.unsqueeze(-1).repeat(1, total_seq)  # [B, S+1]

                # 通过Transformer
                seq_out, gating = self.transfomerlayer(sequence, sequence, sequence,pop_seq=pop_seq)
                seq_out = torch.flatten(seq_out, 1)
            else:
                seq_out = torch.flatten(torch.cat([course_feature, candidate_feature.unsqueeze(1)], dim=1), 1)


            # 最终预测
            features = torch.cat([user_emb, seq_out], dim=1)
            scores=self.sigmoid(self.liner(features))



            return (student,candidate,scores,cl_loss)








class MixtureOfHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, K, h_s, dropout=0.1, bias=True):
        #暂未公开

    def forward(self, x, mask=None, pop_seq=None):
        # 暂未公开


        return out, g  # 返回 g 以便后续计算 load-balance loss


class TransformerLayerMoH(nn.Module):
    def __init__(self, d_model, d_ff, n_heads,  dropout=0.1):

    # 暂未公开

    def forward(self, query, key, value, mask=None, pop_seq=None):
        # 暂未公开
        return x, gating

if __name__ == '__main__':
    pass
