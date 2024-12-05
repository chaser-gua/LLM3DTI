from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import PredModel
import argparse
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Sampler
from dti_dataset import DTIDataset
from utils import set_seed, dict_to_object, weight_init, get_logger
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef, f1_score
from config import cfg
from cl_loss import *
import os
from sklearn.model_selection import StratifiedKFold
import random


"""
ablation对应关系
11：只有cra
12：只有LLM
13：cra和LLM都没有
"""



def get_DTI(dti_path):
    mat_dti = np.loadtxt(dti_path)
    # print(mat_dti.shape)
    index = np.where(mat_dti == 1)  
    dti = np.ones(shape=(len(index[0]), 3))
    dti[:,0] = index[0]
    dti[:,1] = index[1]
    dti = dti.astype(int)
    return dti
  
  
def cross_k_folds(data,k_folds,num1, num2, ratio):
    pos_data = []
    neg_data = []
    
    data_adj = np.zeros(shape=(num1, num2))
    data_adj[data[:,0],data[:,1]] = data[:,2]
    pos_index = np.where(data_adj == 1)

    neg_index = np.where(data_adj == 0)
    result = random.sample(range(0, len(neg_index[0])), len(pos_index[0]) * ratio)
    neg_selected = np.array([neg_index[0][result], neg_index[1][result]])
    count = 0

    kfold = StratifiedKFold(n_splits=k_folds, shuffle=True)
    for  fold, (train, test) in enumerate(kfold.split(pos_index[0], pos_index[1])):
        train_index_X = pos_index[0][train]
        train_index_Y = pos_index[1][train]
        test_index_X = pos_index[0][test]
        test_index_Y = pos_index[1][test]
        test_index = np.array([test_index_X,test_index_Y])
        test_index = torch.tensor(test_index)
        pos_data.append(test_index)

        neg_data.append(torch.tensor(neg_selected[:, count:count+len(test)]))
        count+= len(test)

    return pos_data, neg_data, torch.tensor(np.array(pos_index)), torch.tensor(np.array(neg_selected))
  

def get_train_pos_from_fold(pos_fold,fold_num,device):
    train_pos  = [0]
    for i in range(len(pos_fold)):
        if not isinstance(pos_fold[0], torch.Tensor):
            temp_fold = torch.tensor(pos_fold[i])
        else:
            temp_fold = pos_fold[i]
            
        if i == fold_num:
            continue
        elif len(train_pos) == 1:
            train_pos = temp_fold
        else:
             train_pos = torch.cat((train_pos,temp_fold),dim=-1)
    return train_pos.to(device)
  
  
def gen_train_data(fold_num, pos_fold, neg_fold, drug_num, protein_num,device,ratio):
    all_index = np.arange(drug_num * protein_num)
    test_mask = np.ones(drug_num * protein_num)

    """获得除该fold_num之外的正样本，组合成训练集正样本"""
    pos_index = []
    for i,k in enumerate(pos_fold):
        """从mask中去除正样本数据"""
        index_id = k[0, :] * protein_num + k[1, :]
        test_mask[index_id] = 0

        if i == fold_num:
            continue
        if len(pos_index) == 0:
            pos_index = pos_fold[i]
        else:
            pos_index = np.concatenate((pos_index, pos_fold[i]), axis=-1)

    """
    从mask中去除test集的负样本
    """
    neg_removed = neg_fold[fold_num]
    neg_removed_id = neg_removed[0,:] * protein_num + neg_removed[1,:]
    test_mask[neg_removed_id] = 0

    """
    生成train负样本
    """
    id_n = np.where(test_mask == True)
    num_neg_samples = pos_index.shape[1] * ratio
    id_chosen = np.random.choice(id_n[0], num_neg_samples)
    drug_id = np.floor_divide(id_chosen, protein_num)
    protein_id = np.mod(id_chosen, protein_num)
    neg_index = np.array([drug_id, protein_id])

    index = np.concatenate((pos_index, neg_index), axis=-1)
    pos_labels = np.ones(pos_index.shape[1])
    neg_labels = np.zeros(neg_index.shape[1])
    labels = np.concatenate((pos_labels, neg_labels), axis=-1)

    index = torch.LongTensor(index)
    labels = torch.Tensor(labels)
    shuff_temp = np.arange(0, len(labels))
    random.shuffle(shuff_temp)
    index = index[:, shuff_temp].to(device)
    labels = labels[shuff_temp].to(device)

    return index.to(device), labels.to(device)

  
def train(epoch, train_index, train_label, features, model, opt, loss_func, logging):
    model.train()
    opt.zero_grad()
    dt_features, pred = model((features[0][train_index[0,:],:], features[1][train_index[1,:],:]))
    loss = loss_func(pred, train_label.long())
    loss.backward()
    opt.step()
    if (epoch+1) % 10 == 0:
        logging.info('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))
    else:
        print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, loss.item()))


def model_test(test_index, test_label, features, model, logging):
    model.eval()
    with torch.no_grad():
        dt_features, pred = model((features[0][test_index[0,:],:], features[1][test_index[1,:],:]))
        pred_prob = F.softmax(pred, dim=1)
        pred_label = torch.argmax(pred_prob, dim=1)

        ac = accuracy_score(test_label.cpu(), pred_label.cpu())
        auroc = roc_auc_score(test_label.cpu(), pred_prob[:, 1].cpu())
        precision, recall, thresholds = precision_recall_curve(test_label.cpu(), pred_prob[:, 1].cpu())
        aupr = auc(recall, precision)
        mcc = matthews_corrcoef(test_label.cpu(), pred_label.cpu())
        f1 = f1_score(test_label.cpu(), pred_label.cpu())

        logging.info('*********ACC:{:.4f} | AUROC:{:.4f} | AUPR:{:.4f} | MCC:{:.4f} | F1:{:.4f}*********'.format(ac, auroc, aupr, mcc, f1))
        
        return ac, auroc, aupr, mcc, f1
      

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--epochs", type=int, default=500)
  parser.add_argument("--mode", type=str, default='sl')
  parser.add_argument("--ratio", type=int, default=1)
  parser.add_argument("--method", type=str, default='fold')
  args = parser.parse_args()
  
  seed = args.seed
  device = args.device
  max_epochs = args.epochs
  train_mode = args.mode
  ratio = args.ratio
  method = args.method
  
  set_seed(seed)
  
  log_path = "./log/fold/others/"
  logging = get_logger(log_dir=log_path)
  
  logging.info(f"seed: {seed}, max_epochs: {max_epochs}, train_mode: {train_mode}, ratio: 1:{ratio}, method: {method}")
  
  dti = get_DTI("../data/mat_drug_protein.txt")
  
  dti_pos_fold, dti_neg_data_fold, pos, neg = cross_k_folds(dti, 5, 708, 1493, ratio)
  
  protein_ontological = torch.tensor(np.loadtxt('../data/protein_vector_unique.txt'))
  protein_similarity = torch.tensor(np.loadtxt('../data/protein_similarity_unique.txt'))
  protein_llm_embeddings = torch.tensor(np.load('../data/protein_embeddings_final.npy'))
  drug_ontological = torch.tensor(np.loadtxt('../data/drug_vector_d100.txt'))
  drug_similarity = torch.tensor(np.loadtxt('../data/drug_similarity.txt'))
  drug_llm_embeddings = torch.tensor(np.load('../data/drug_embeddings_sorted.npy'))


  # ===whole exp=== #
  drug_features = torch.cat((drug_ontological, drug_similarity, drug_llm_embeddings),
                                        dim=1).float().to(device)
  prot_features = torch.cat((protein_ontological, protein_similarity, protein_llm_embeddings),
                                        dim=1).float().to(device)
  
  
  # ===ablation1_1=== #
  drug_features = torch.cat((drug_ontological, drug_similarity),
                                        dim=1).float().to(device)
  prot_features = torch.cat((protein_ontological, protein_similarity),
                                        dim=1).float().to(device)  
  # ===ablation1_1=== #


#   # ===ablation1_12=== #
#   drug_features = drug_llm_embeddings.float().to(device)
#   prot_features = protein_llm_embeddings.float().to(device)  
#   # ===ablation1_1=== #
  
  
  features = (drug_features, prot_features)
  
  model = PredModel(dict_to_object(cfg)).to(device)  # model的forward传入features
  model.apply(weight_init)
  opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
  
  loss_func = nn.CrossEntropyLoss()
  
  test_results = []
  for fold_num in range(5):
      dti_test_index = torch.cat((dti_pos_fold[fold_num], dti_neg_data_fold[fold_num]), dim=1).long()
      dti_test_labels = torch.cat(
              (torch.ones_like(dti_pos_fold[fold_num][0, :]), torch.zeros_like(dti_neg_data_fold[fold_num][0, :])),
              dim=0).long()
      shuff_temp = np.arange(0, len(dti_test_labels))
      random.shuffle(shuff_temp)
      dti_test_index = dti_test_index[:, shuff_temp].to(device)
      dti_test_labels = dti_test_labels[shuff_temp].to(device)
      
      dti_train_pos = get_train_pos_from_fold(dti_pos_fold, fold_num, device)
      
      for epoch in range(max_epochs):
          dti_train_index, dti_train_label = gen_train_data(fold_num, dti_pos_fold, dti_neg_data_fold, 708, 1493, device, ratio)
          train(epoch, dti_train_index, dti_train_label, features, model, opt, loss_func, logging)
      print("****************************************")
      ac, auroc, aupr, mcc, f1 = model_test(dti_test_index, dti_test_labels, features, model, logging)
      test_results.append({'ac': ac, 'auroc': auroc, 'aupr': aupr, 'mcc': mcc, 'f1': f1})

  # 计算平均值
  avg_results = {}
  for metric in ['ac', 'auroc', 'aupr', 'mcc', 'f1']:
      avg_results[metric] = np.mean([result[metric] for result in test_results])

  # 打印平均结果
  print("****************************************")
  print("平均结果为:")
  for metric, value in avg_results.items():
      print(f"{metric.upper()}:{value:.4f} ", end="")  
      
  print()
      
  logging.info(f"5-fold average results: {avg_results}")
  