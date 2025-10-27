import math
import numpy as np
import heapq
import matplotlib.pyplot as plt
import pandas as pd
from torch import nn
from tqdm import tqdm
import torch
from sklearn.metrics import roc_auc_score
from BPR_Loss import bpr_loss

def eva_rating(candidate_list, predict_list, target_list):
    data_size = len(candidate_list)
    map_item_predict = {candidate_list[i]: predict_list[i] for i in range(data_size)}

    rank_list_1 = heapq.nlargest(1, map_item_predict, key=map_item_predict.get)
    rank_list_5 = heapq.nlargest(5, map_item_predict, key=map_item_predict.get)
    rank_list_10 = heapq.nlargest(10, map_item_predict, key=map_item_predict.get)
    rank_list_all = heapq.nlargest(data_size, map_item_predict, key=map_item_predict.get)

    hr1 = HitRatio(rank_list_1, target_list)
    hr5 = HitRatio(rank_list_5, target_list)
    ndcg5 = NDCG(rank_list_5, target_list)
    hr10 = HitRatio(rank_list_10, target_list)
    ndcg10 = NDCG(rank_list_10, target_list)
    mrr = MRR(rank_list_all, target_list)

    return hr1, hr5, ndcg5, hr10, ndcg10, mrr

# calculate metrics
def HitRatio(rank_list, target):
    for item in rank_list:
        if item in target:
            return 1
    return 0


def MRR(rank_list, target):
    for index, item in enumerate(rank_list):
        if item in target:
            return 1.0 / (index + 1.0)
    return 0


def NDCG(rank_list, target):
    for i in range(len(rank_list)):
        if rank_list[i] in target:
            return math.log(2) / math.log(i + 2)
    return 0


# plot loss, auc and metrics
def plot(title, xticks, values, labels):
    plt.figure(num=1, figsize=(8, 6))
    for i, value in enumerate(values):
        plt.plot(xticks, value, label=labels[i])
    plt.legend()
    plt.grid(axis='both')
    plt.xlabel('epoch')
    plt.ylabel('value')
    plt.title(title)
    plt.show()



def evaluation(loader, model, device):
    model.eval()
    student_array = np.array([])
    predict_array = np.array([])
    candidate_array = np.array([], dtype=int)
    label_array = np.array([], dtype=int)
    pop_array = np.array([], dtype=int)


    total_loss = 0.0
    count = 0
    loss_fn = nn.BCELoss(reduction='mean').to(device)

    bpr_loss_list = []  # 新增：记录每组 BPR loss

    print('start testing...')
    with torch.no_grad():
        for inputs in tqdm(loader, desc='testing'):
            (student, courses, category, candidate, candidate_cate, label,hist_len,pop_dict) = inputs
            label = label.float()

            if device == 'cuda:0':
                student = student.cuda()
                courses = courses.cuda()
                category = category.cuda()
                candidate = candidate.cuda()
                candidate_cate = candidate_cate.cuda()
                label = label.cuda()
                hist_len = hist_len.cuda()
                pop_dict= pop_dict.cuda()

            student, candidate, y_hat, cl_loss= model(student, courses, category, candidate, candidate_cate, False,pop_dict=pop_dict)

            # Loss 和训练阶段一致
            loss = cl_loss
            total_loss += loss.item()
            count += 1

            # 保存预测结果和label
            student_array= np.concatenate((student_array, student.cpu().flatten().numpy()))
            predict_array = np.concatenate((predict_array, y_hat.cpu().flatten().numpy()))
            candidate_array = np.concatenate((candidate_array, candidate.cpu().numpy()))
            label_array = np.concatenate((label_array, label.cpu().numpy()))
            pop_array = np.concatenate((pop_array, pop_dict.cpu().numpy().reshape(-1)))  # 展平记录


    # 排名相关评估（100条一组）
    candidate_list = candidate_array.reshape(-1, 100)
    predict_list = predict_array.reshape(-1, 100)
    target_list = candidate_list[:, :1]
    pop_list = pop_array.reshape(-1, 100)  # 对齐每组的 pop_dict


    hits_1, hits_5, ndcgs_5, hits_10, ndcgs_10, mrrs = [], [], [], [], [], []
    # 新增：长尾项指标列表
    tail_hits_1, tail_hits_5, tail_ndcgs_5, tail_hits_10, tail_ndcgs_10 = [], [], [], [], []
    hot_hits_1, hot_hits_5, hot_ndcgs_5, hot_hits_10, hot_ndcgs_10 = [], [], [], [], []


    for i in range(len(candidate_list)):
        hr1, hr5, ndcg5, hr10, ndcg10, mrr = eva_rating(candidate_list[i], predict_list[i], target_list[i])
        hits_1.append(hr1)
        hits_5.append(hr5)
        ndcgs_5.append(ndcg5)
        hits_10.append(hr10)
        ndcgs_10.append(ndcg10)
        mrrs.append(mrr)

        # ===== 计算 BPR loss =====
        pos_score = predict_list[i][0]  # 正样本分数
        neg_scores = predict_list[i][1:]  # 负样本分数

        # 转为 tensor
        pos_score_tensor = torch.tensor([pos_score], device='cpu')
        neg_scores_tensor = torch.tensor(neg_scores, device='cpu')

        # 调用你的 bpr_loss 函数
        bpr_loss_val = bpr_loss(pos_score_tensor, neg_scores_tensor)
        bpr_loss_list.append(bpr_loss_val.item())

        # 如果正样本是长尾课程（pop_dict == -1），则记录其指标
        if pop_list[i][0] == -1:
            tail_hits_1.append(hr1)
            tail_hits_5.append(hr5)
            tail_ndcgs_5.append(ndcg5)
            tail_hits_10.append(hr10)
            tail_ndcgs_10.append(ndcg10)
            # 如果该组为 few-shot（hist_len <= 5），则记录 few-shot 指标

        if pop_list[i][0] == 1:
            hot_hits_1.append(hr1)
            hot_hits_5.append(hr5)
            hot_ndcgs_5.append(ndcg5)
            hot_hits_10.append(hr10)
            hot_ndcgs_10.append(ndcg10)


    # AUC 单独计算
    auc = roc_auc_score(label_array, predict_array)

    # 排名结果
    hr1_result = np.array(hits_1).mean()
    hr5_result = np.array(hits_5).mean()
    ndcg5_result = np.array(ndcgs_5).mean()
    hr10_result = np.array(hits_10).mean()
    ndcg10_result = np.array(ndcgs_10).mean()
    mrr_result = np.array(mrrs).mean()

    # 平均 Loss
    avg_loss = total_loss / count+np.mean(bpr_loss_list)

    # 长尾课程指标（若无长尾样本，则结果为 0）
    tail_hr1 = np.mean(tail_hits_1) if tail_hits_1 else 0.0
    tail_hr5 = np.mean(tail_hits_5) if tail_hits_5 else 0.0
    tail_ndcg5 = np.mean(tail_ndcgs_5) if tail_ndcgs_5 else 0.0
    tail_hr10 = np.mean(tail_hits_10) if tail_hits_10 else 0.0
    tail_ndcg10 = np.mean(tail_ndcgs_10) if tail_ndcgs_10 else 0.0

    # 热门课程指标（若无热门样本，则结果为 0）
    hot_hr1 = np.mean(hot_hits_1) if hot_hits_1 else 0.0
    hot_hr5 = np.mean(hot_hits_5) if hot_hits_5 else 0.0
    hot_ndcg5 = np.mean(hot_ndcgs_5) if hot_ndcgs_5 else 0.0
    hot_hr10 = np.mean(hot_hits_10) if hot_hits_10 else 0.0
    hot_ndcg10 = np.mean(hot_ndcgs_10) if hot_ndcgs_10 else 0.0



    result_df = pd.DataFrame({
        'student': student_array,
        'candidate_id': candidate_array,
        'score': predict_array,
        'label': label_array,
        'pop':pop_array
    })

    # 可自定义文件名路径
    print('保存测试结果中...')
    result_df.to_csv('/GraduateStudents/chenyuewen/RecommenderSystem/CLGADN_V18/data/courses/test_scores.csv', index=False)
    print('测试结果已保存。')
    # 返回增加 avg_bpr_loss
    return auc, hr1_result, hr5_result, ndcg5_result, hr10_result, ndcg10_result, mrr_result, avg_loss,tail_hr1, tail_hr5, tail_ndcg5, tail_hr10, tail_ndcg10, hot_hr1, hot_hr5, hot_ndcg5, hot_hr10, hot_ndcg10
