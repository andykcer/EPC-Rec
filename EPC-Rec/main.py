from load_data import load_data
from EPC_Rec import FAME
import torch.nn as nn
from Evaluation import *
from Parse import parse_mooc_args
from Utils import seed_everthing
import os
from log_helper import *
from helper import *
import numpy as np
import swanlab
import logging
from BPR_Loss import bpr_loss
logging.getLogger("urllib3").setLevel(logging.WARNING)

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# torch.multiprocessing.set_sharing_strategy('file_system')
# import torch.multiprocessing
# torch.multiprocessing.set_sharing_strategy('file_system')

if __name__ == '__main__':

    args = parse_mooc_args()

    swanlab.init(
        # 设置将记录此次运行的项目信息
        project="CLGADN_V18",
        workspace="andykcer",
        # 新的里程
        config={
            "learning_rate": args.lr,
            "architecture": "LightGCN-Transfomer",
            "experiment_name":"Listwise+hardsample（1/4）+MOH（10*listwiseloss+CLloss+5lb）",
            "dataset": "MOOC",
            "epochs": args.epochs
        }
    )

    args.quick_test = False

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(log_save_id), no_console=False)
    # logging.info(args)

    seed_everthing(args.seed)

    logging.info('quick_test:{} and transformer:{}'.format(args.quick_test, args.transformer))




    train_loader, test_loader, Graph1, Graph2 = load_data(args)







    args.device = 'cpu'
    if args.use_cuda and torch.cuda.is_available():
        logging.info('cuda ready...')
        args.device = 'cuda:0'
    else:
        logging.info('cpu ready...')
    logging.info(args)

    model = FAME(args, Graph1, Graph2)
    # logging.info(model)

    # print("===================== model location:", next(model.parameters()).device)

    # print(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    for epoch in range(args.epochs):
        total_loss_epoch = 0.0
        total_tmp = 0
        model.train()
        logging.info('***************************************')
        logging.info('epoch:{}_start training...'.format(epoch))
        for inputs in tqdm(train_loader, desc='training'):
            # print(x.shape,x.dtype)
            (student, courses, category, candidate, candidate_cate, label,neg_candidate,neg_category,pop_dict) = inputs

            label = label.float()
            # print('data to', device, 'done!')
            if args.device == 'cuda:0':
                student = student.cuda()

                courses = courses.cuda()
                category = category.cuda()

                candidate = candidate.cuda()
                candidate_cate = candidate_cate.cuda()
                neg_candidate = neg_candidate.cuda()
                neg_category = neg_category.cuda()
                pop_dict=pop_dict.cuda()

                label = label.cuda()
            # print("==================data location:", student.device)
            student, candidate, y_hat, loss= model(student, courses, category, candidate, candidate_cate,True,neg_candidate,neg_category,pop_dict)

            optimizer.zero_grad()



            loss.backward()
            optimizer.step()
            total_loss_epoch += loss.item()
            total_tmp += 1
        avg_loss = total_loss_epoch / total_tmp
        logging.info(f'Epoch {epoch} average loss: {avg_loss:.6f}')

        auc, hits_1, hits_5, ndcgs_5, hits_10, ndcgs_10, mrrs, test_loss,tail_hr1, tail_hr5, tail_ndcg5, tail_hr10, tail_ndcg10 , hot_hr1, hot_hr5, hot_ndcg5, hot_hr10, hot_ndcg10= evaluation(test_loader, model, args.device)

        test_msg = (
            f'epoch: {epoch}    test loss: {test_loss:.6f}    test auc: {auc:.4f}    '
            f'hits@1: {hits_1:.4f}    hits@5: {hits_5:.4f}    '
            f'ndcg@5: {ndcgs_5:.4f}    hits@10: {hits_10:.4f}    '
            f'ndcg@10: {ndcgs_10:.4f}    MRR: {mrrs:.4f}'
            f' tail_hr1: {tail_hr1:.4f}    tail_hr5: {tail_hr5:.4f}    '
            f'tail_ndcg5: {tail_ndcg5:.4f}    tail_hr10: {tail_hr10:.4f}    '
            f'tail_ndcg10: {tail_ndcg10:.4f}'
            f' hot_hr1: {hot_hr1:.4f}    hot_hr5: {hot_hr5:.4f}    '
            f'hot_ndcg5: {hot_ndcg5:.4f}    hot_hr10: {hot_hr10:.4f}    '
            f'hot_ndcg10: {hot_ndcg10:.4f}'

        )
        logging.info(test_msg)
        swanlab.log({
            "epoch": epoch,
            "train loss": round(avg_loss, 6),
            "Test loss": round(test_loss, 6),
            "Test auc": round(auc, 4),
            "hits@1": round(hits_1, 4),
            "hits@5": round(hits_5, 4),
            "hits@10": round(hits_10, 4),
            "ndcg@5": round(ndcgs_5, 4),
            "ndcg@10": round(ndcgs_10, 4),
            "MRR": round(mrrs, 4),
            "tail_hr1": round(tail_hr1, 4),
            "tail_hr5": round(tail_hr5, 4),
            "tail_ndcg5": round(tail_ndcg5, 4),
            "tail_hr10": round(tail_hr10, 4),
            "tail_ndcg10": round(tail_ndcg10, 4),



        })

        # 效果最好，直接保存然后break了

    # os.system("shutdown")
