from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from model import LLM3Model
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
import statistics
# 导入需要的函数
from sklearn.metrics import precision_score, recall_score

def calculate_aupr(labels, preds):
  precision, recall, _ = precision_recall_curve(labels, preds)
  return auc(recall, precision)


class BalancedBatchSampler(Sampler):
    def __init__(self, positive_indices, negative_indices, batch_size):
        self.positive_indices = positive_indices
        self.negative_indices = negative_indices
        self.batch_size = batch_size // 2  # 每个batch中正负样本数量相等
        self.num_batches = min(len(positive_indices), len(negative_indices)) // self.batch_size

    def __iter__(self):
        positive_indices = np.random.permutation(self.positive_indices)
        negative_indices = np.random.permutation(self.negative_indices)

        for i in range(self.num_batches):
            pos_batch = positive_indices[i * self.batch_size: (i + 1) * self.batch_size]
            neg_batch = negative_indices[i * self.batch_size: (i + 1) * self.batch_size]
            batch_indices = np.concatenate((pos_batch, neg_batch))
            np.random.shuffle(batch_indices)
            yield batch_indices

    def __len__(self):
        return self.num_batches


class RatioBatchSampler(Sampler):
  def __init__(self, positive_indices, negative_indices, batch_size, ratio=1):
    self.positive_indices = positive_indices
    self.negative_indices = negative_indices
    self.pos_batch_size = batch_size // (1 + ratio)  # 每个batch中的正样本数量
    self.neg_batch_size = self.pos_batch_size * ratio  # 每个batch中的负样本数量
    self.num_batches = min(len(positive_indices) // self.pos_batch_size,
                            len(negative_indices) // self.neg_batch_size)
    self.ratio = ratio

  def __iter__(self):
    positive_indices = np.random.permutation(self.positive_indices)
    negative_indices = np.random.permutation(self.negative_indices)

    for i in range(self.num_batches):

      pos_batch = positive_indices[i * self.pos_batch_size: (i + 1) * self.pos_batch_size]
      neg_batch = negative_indices[i * self.neg_batch_size: (i + 1) * self.neg_batch_size]

      batch_indices = np.concatenate((pos_batch, neg_batch))
      np.random.shuffle(batch_indices)
      yield batch_indices

  def __len__(self):
    return self.num_batches

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", type=str, default="cuda")
  parser.add_argument("--epochs", type=int, default=100)
  parser.add_argument("--mode", type=str, default='sl')
  parser.add_argument("--ratio", type=int, default=1)
  parser.add_argument("--note", type=str, default="")
  args = parser.parse_args()
  
  seed = args.seed
  device = args.device
  max_epochs = args.epochs
  train_mode = args.mode
  ratio = args.ratio
  note = args.note
  
  log_path = "./log/llmmm/"
  logging = get_logger(log_dir=log_path)
  
  acc5 = []
  auroc5 = []
  aupr5 = []
  mcc5 = []
  f15 = []
  
  for seed in range(5):
  
    set_seed(seed)
    
    logging.info(f"seed: {seed}, max_epochs: {max_epochs}, train_mode: {train_mode}, ratio: 1:{ratio}, note: {note}")
    
    protein_ontological = np.loadtxt('../data/protein_vector_unique.txt')
    protein_similarity = np.loadtxt('../data/protein_similarity_unique.txt')
    protein_llm_embeddings = np.load('../data/protein_embeddings_final.npy')
    drug_ontological = np.loadtxt('../data/drug_vector_d100.txt')
    drug_similarity = np.loadtxt('../data/drug_similarity.txt')
    drug_llm_embeddings = np.load('../data/drug_embeddings_sorted.npy')
    protein_drug_matrix = np.loadtxt('../data/mat_protein_drug_unique.txt', dtype=int)
    protein_order = np.loadtxt('../data/protein_unique.txt', dtype=str)
    drug_order = np.loadtxt('../data/drug.txt', dtype=str)
    
    
    # """ablation text smile"""
    # drug_llm_embeddings = np.load('../data/morgan.npy')
    # protein_llm_embeddings = np.load('../data/kmer.npy')
    
    # print(drug_llm_embeddings.shape)
    # print(protein_llm_embeddings.shape)
    
    # exit(0)
    
    ## prepare data
    dataset = DTIDataset(protein_ontological, protein_similarity, protein_llm_embeddings,
                        drug_ontological, drug_similarity, drug_llm_embeddings,
                        protein_drug_matrix, protein_order, drug_order)

    # Separate positive and negative samples
    positive_indices = [i for i in range(len(dataset)) if dataset[i][-1] == 1]
    negative_indices = [i for i in range(len(dataset)) if dataset[i][-1] == 0]

    # Ensure that the number of positive and negative samples are equal
    min_len = min(len(positive_indices), len(negative_indices))

    positive_indices = np.random.choice(positive_indices, min_len, replace=False).tolist()
    negative_indices = np.random.choice(negative_indices, min_len * ratio, replace=False).tolist()

    # Combine positive and negative indices
    balanced_indices = positive_indices + negative_indices

    # Split into training, validation, and test sets (keeping 1:1 ratio)
    train_indices, test_indices = train_test_split(balanced_indices, test_size=0.2, random_state=42)
    train_indices, val_indices = train_test_split(train_indices, test_size=0.1, random_state=42)
    
    # === ratio sampler === #
    train_sampler = RatioBatchSampler([idx for idx in train_indices if idx in positive_indices],
                                      [idx for idx in train_indices if idx in negative_indices], batch_size=64, ratio=ratio)
    val_sampler = RatioBatchSampler([idx for idx in val_indices if idx in positive_indices],
                                    [idx for idx in val_indices if idx in negative_indices], batch_size=64, ratio=ratio)
    test_sampler = RatioBatchSampler([idx for idx in test_indices if idx in positive_indices],
                                      [idx for idx in test_indices if idx in negative_indices], batch_size=64, ratio=ratio)

    train_loader = DataLoader(dataset, batch_sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_sampler=val_sampler)
    test_loader = DataLoader(dataset, batch_sampler=test_sampler)

    model = LLM3Model(dict_to_object(cfg)).to(device)
    model.apply(weight_init)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['lr'], weight_decay=cfg['wd'])
    
    ## Train and Valid
    max_acc = 0
    best_epoch = 0
    for epoch in range(max_epochs):
      step = 0
      epoch_loss = 0
      f1_train = 0
      acc_train = 0
      auroc_train = 0
      aupr_train = 0
      mcc_train = 0
      # recal_train = 0
      # pre_train = 0
      model.train()
      for batch_idx, train_data in enumerate(train_loader):
        step += 1
        opt.zero_grad() 
        prot_ontological_feature, drug_ontological_feature, \
        prot_similarity_feature, drug_similarity_feature, \
        prot_llm_embedding, drug_llm_embedding, labels = train_data   #, label_features
        
        # drug_features = torch.cat((drug_ontological_feature, 
        #                            drug_similarity_feature, 
        #                            drug_llm_embedding
        #                            ), dim=1).float().to(device)
        # prot_features = torch.cat((prot_ontological_feature, 
        #                            prot_similarity_feature, 
        #                            prot_llm_embedding
        #                            ), dim=1).float().to(device)
        
        drug_stru_features = torch.cat((drug_ontological_feature, 
                                  drug_similarity_feature), dim=1).float().to(device)  
        prot_stru_features = torch.cat((prot_ontological_feature, 
                                  prot_similarity_feature), dim=1).float().to(device) 
        
        drug_llm_features = drug_llm_embedding.float().to(device)
        prot_llm_features = prot_llm_embedding.float().to(device)
        
        # """ablation text"""
        # drug_llm_features = torch.rand(drug_llm_features.shape[0], drug_llm_features.shape[1]).to(device)
        # prot_llm_features = torch.rand(prot_llm_features.shape[0], prot_llm_features.shape[1]).to(device)

        # print(drug_features.shape)
        # print(prot_features.shape)

        labels = labels.long().to(device)
        # label_features = label_features.float().to(device)
          
        dt_features, pred = model((drug_stru_features, drug_llm_features, prot_stru_features, prot_llm_features))
        
        if train_mode == 'sl':
          loss = nn.CrossEntropyLoss()(pred, labels)
        else:
          loss = get_match_loss(dt_features, label_features, model, device, cfg['bs'])
          
        loss.backward()
        opt.step()
        
        outputs_prob = torch.softmax(pred, dim=1)[:, 1].cpu().detach().numpy()
        outputs_class = np.argmax(pred.cpu().detach().numpy(), axis=1)
        acc_value = accuracy_score(labels.cpu().detach().numpy(), outputs_class)
        f1_value = f1_score(labels.cpu().detach().numpy(), outputs_class)
        aupr_value = calculate_aupr(labels.cpu().detach().numpy(), outputs_class)
        mcc_value = matthews_corrcoef(labels.cpu().detach().numpy(), outputs_class)
        auroc_value = roc_auc_score(labels.cpu().detach().numpy(), outputs_prob)
        # precision_value = precision_score(labels_np, outputs_class)  # 精确率
        # recall_value = recall_score(labels_np, outputs_class)        # 召回率
        
        epoch_loss += loss.item()
        acc_train += acc_value
        auroc_train += auroc_value
        aupr_train += aupr_value
        mcc_train += mcc_value
        f1_train += f1_value    
        
      epoch_loss /= step
      acc_train /= step
      auroc_train /= step
      aupr_train /= step
      mcc_train /= step
      f1_train /= step
      # logging.info(f"Train | epoch={epoch} | loss={epoch_loss:.4f}, auc_pr={auc_pr_train:.4f}, f1={f1_train:.4f}")
      logging.info(f"Train | epoch={epoch} | loss={epoch_loss:.4f}, acc={acc_train:.4f}, auroc={auroc_train:.4f}, aupr={aupr_train:.4f}, mcc={mcc_train:.4f}, f1={f1_train:.4f}")


      step = 0
      acc_val = 0
      auroc_val = 0
      aupr_val = 0
      mcc_val = 0
      f1_val = 0
      model.eval()
      for batch_idx, val_data in enumerate(val_loader):
        step += 1
        prot_ontological_feature, drug_ontological_feature, \
        prot_similarity_feature, drug_similarity_feature, \
        prot_llm_embedding, drug_llm_embedding, labels = val_data      #, label_features
        
        drug_stru_features = torch.cat((drug_ontological_feature, 
                                  drug_similarity_feature), dim=1).float().to(device)  
        prot_stru_features = torch.cat((prot_ontological_feature, 
                                  prot_similarity_feature), dim=1).float().to(device) 
        
        drug_llm_features = drug_llm_embedding.float().to(device)
        prot_llm_features = prot_llm_embedding.float().to(device)

        # print(drug_features.shape)
        # print(prot_features.shape)

        labels = labels.long().to(device)
        # label_features = label_features.float().to(device)
          
        dt_features, pred = model((drug_stru_features, drug_llm_features, prot_stru_features, prot_llm_features))
          
        outputs_prob = torch.softmax(pred, dim=1)[:, 1].cpu().detach().numpy()
        outputs_class = np.argmax(pred.cpu().detach().numpy(), axis=1)
        acc_val += accuracy_score(labels.cpu().detach().numpy(), outputs_class)
        auroc_val += roc_auc_score(labels.cpu().detach().numpy(), outputs_prob)
        aupr_val += calculate_aupr(labels.cpu().detach().numpy(), outputs_prob)
        mcc_val += matthews_corrcoef(labels.cpu().detach().numpy(), outputs_class)
        f1_val += f1_score(labels.cpu().detach().numpy(), outputs_class)
        
        
      acc_val /= step
      auroc_val /= step
      aupr_val /= step
      mcc_val /= step
      f1_val /= step
      # logging.info(f"Valid | epoch={epoch} | auc_pr={auc_val:.4f}, f1={f1_val:.4f}")  
      logging.info(f"Valid | epoch={epoch} | acc={acc_val:.4f}, auroc={auroc_val:.4f}, aupr={aupr_val:.4f}, mcc={mcc_val:.4f}, f1={f1_val:.4f}")
      
      cur_acc = statistics.mean([acc_val, auroc_val, aupr_val, mcc_val, f1_val])
      
      os.makedirs("./result/llmmm/", exist_ok=True)
      if cur_acc > max_acc:
        best_epoch = epoch
        save_dir = f"./result/llmmm/epoch_{best_epoch}_{ratio}.pth"
        torch.save(model, save_dir)
        max_acc = cur_acc
          
    logging.info(f"best epoch: {best_epoch}")
          
    for filename in os.listdir("./result/llmmm/"):
      if not filename.startswith(f"epoch_{best_epoch}") and filename.endswith(f"{ratio}.pth"):
        os.remove(f"./result/llmmm/{filename}")
    
    
    ## Test
    model = LLM3Model(dict_to_object(cfg)).to(device)
    model = torch.load(save_dir)
      
    step = 0
    acc_test = 0
    auroc_test = 0
    aupr_test = 0
    mcc_test = 0
    f1_test = 0
    model.eval()
    for batch_idx, test_data in enumerate(test_loader):
      step += 1
      prot_ontological_feature, drug_ontological_feature, \
      prot_similarity_feature, drug_similarity_feature, \
      prot_llm_embedding, drug_llm_embedding, labels = test_data      
    
      drug_stru_features = torch.cat((drug_ontological_feature, 
                                  drug_similarity_feature), dim=1).float().to(device)  
      prot_stru_features = torch.cat((prot_ontological_feature, 
                                  prot_similarity_feature), dim=1).float().to(device) 
      
      drug_llm_features = drug_llm_embedding.float().to(device)
      prot_llm_features = prot_llm_embedding.float().to(device)

      # print(drug_features.shape)
      # print(prot_features.shape)

      labels = labels.long().to(device)
      # label_features = label_features.float().to(device)
        
      dt_features, pred = model((drug_stru_features, drug_llm_features, prot_stru_features, prot_llm_features))
      
      outputs_prob = torch.softmax(pred, dim=1)[:, 1].cpu().detach().numpy()
      outputs_class = np.argmax(pred.cpu().detach().numpy(), axis=1)
      acc_test += accuracy_score(labels.cpu().detach().numpy(), outputs_class)
      auroc_test += roc_auc_score(labels.cpu().detach().numpy(), outputs_prob)
      aupr_test += calculate_aupr(labels.cpu().detach().numpy(), outputs_prob)
      mcc_test += matthews_corrcoef(labels.cpu().detach().numpy(), outputs_class)
      f1_test += f1_score(labels.cpu().detach().numpy(), outputs_class)
      
      
    acc_test /= step
    auroc_test /= step
    aupr_test /= step
    mcc_test /= step
    f1_test /= step
    logging.info(f"Test | acc={acc_test:.4f}, auroc={auroc_test:.4f}, aupr={aupr_test:.4f}, mcc={mcc_test:.4f}, f1={f1_test:.4f}") 

    acc5.append(acc_test)
    auroc5.append(auroc_test)
    aupr5.append(aupr_test)
    mcc5.append(mcc_test)
    f15.append(f1_test)
    
    # exit(0)
    
  logging.info("---------------------------------------")  
  logging.info(f"Test | acc={statistics.mean(acc5):.4f}, auroc={statistics.mean(auroc5):.4f}, aupr={statistics.mean(aupr5):.4f}, mcc={statistics.mean(mcc5):.4f}, f1={statistics.mean(f15):.4f}") 
  logging.info(f"Test | acc={statistics.stdev(acc5):.3f}, auroc={statistics.stdev(auroc5):.3f}, aupr={statistics.stdev(aupr5):.3f}, mcc={statistics.stdev(mcc5):.3f}, f1={statistics.stdev(f15):.3f}") 
  