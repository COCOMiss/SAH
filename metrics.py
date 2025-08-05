
import numpy as np
import torch


def hit_k(y_pred, y_true, k):
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        return 1
    else:
        return 0


def ndcg_k(y_pred, y_true, k):
    y_pred_indices = y_pred.topk(k=k).indices.tolist()
    if y_true in y_pred_indices:
        position = y_pred_indices.index(y_true) + 1
        return 1 / np.log2(1 + position)
    else:
        return 0


def mAP_metric(y_true_seq, y_pred_seq, k):
    rlt = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        rec_list = y_pred.argsort()[-k:][::-1]
        r_idx = np.where(rec_list == y_true)[0]
        if len(r_idx) != 0:
            rlt += 1 / (r_idx[0] + 1)
    return rlt / len(y_true_seq)



def MRR_metric(y_pred_seq,y_true_seq):
    batch_size = y_pred_seq.size(0)
    if batch_size==0:
        return None
    rlt = 0
    count = 0
    for y_true, y_pred in zip(y_true_seq, y_pred_seq):
        if y_true == -1:
            continue
        rec_list=torch.flip(y_pred.argsort(), [0])
        r_idx = np.where(rec_list == y_true)[0][0]
        rlt += 1 / (r_idx + 1)
        count += 1
    return rlt / count



def batch_performance(batch_y_pred, batch_y_true, k):
    batch_size = batch_y_pred.size(0)
    if batch_size==0:
        return None
    batch_recall = 0
    for idx in range(batch_size):
        hit = hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_recall += hit
    recall = batch_recall / batch_size

    return recall





def batch_performance_(batch_y_pred, batch_y_true, k):
    batch_size = batch_y_pred.size(0)
    if batch_size==0:
        return None,None
    batch_recall = 0
    batch_ndcg = 0
    for idx in range(batch_size):
        hit = hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_recall += hit
        ndcg = ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
        batch_ndcg += ndcg
    recall = batch_recall / batch_size
    ndcg = batch_ndcg / batch_size
    
    return recall, ndcg


def user_performance(batch_y_pred, batch_y_true,user_idx, k):
    batch_size = batch_y_pred.size(0)
    user_idx_result={}
    for idx in range(batch_size):
        hit = hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        ndcg = ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
        user_idx_result[user_idx[idx]]=(hit,ndcg)

    return user_idx_result


def sample_performance(batch_y_pred, batch_y_true, k):
    batch_size = batch_y_pred.size(0)
    sample_result=[]
    sample_ndcg=[]
    for idx in range(batch_size):
        hit = hit_k(batch_y_pred[idx], batch_y_true[idx], k)
        ndcg = ndcg_k(batch_y_pred[idx], batch_y_true[idx], k)
        sample_result.append(hit)
        sample_ndcg.append(ndcg)

    return sample_result,sample_ndcg




