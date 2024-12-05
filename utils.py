import torch
import numpy as np
import random
import torch.nn.init as init
import torch.nn as nn
import logging
import os
import sys
from datetime import datetime

def set_seed(seed):
  torch.manual_seed(seed)  
  torch.cuda.manual_seed_all(seed)
  np.random.seed(seed)
  random.seed(seed)
  torch.backends.cudnn.deterministic = True
  

class Dict(dict):
  __setattr__ = dict.__setitem__
  __getattr__ = dict.__getitem__

  
def dict_to_object(dictObj):
  if not isinstance(dictObj, dict):
    return dictObj
  inst=Dict()
  for k,v in dictObj.items():
    inst[k] = dict_to_object(v)
  return inst


# 初始化工具
def weight_init(m):
    """
    Usage:
        model = Model()
        model.apply(weight_init)
    """   
    if isinstance(m, nn.Conv1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        init.normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        init.normal_(m.weight.data, mean=1, std=0.02)
        init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        init.xavier_normal_(m.weight.data)
        # init.normal_(m.bias.data)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                init.orthogonal_(param.data)
            else:
                init.normal_(param.data)
    elif isinstance(m, nn.Embedding):
        embed_size = m.weight.size(-1)
        if embed_size > 0:
            init_range = 0.5 / m.weight.size(-1)
            init.uniform_(m.weight.data, -init_range, init_range)
    elif isinstance(m, nn.Bilinear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
            
            
def get_logger(log_dir):
  logger = logging.getLogger()
  logger.setLevel(logging.INFO)

  current_time = datetime.now()
  formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
  os.makedirs(log_dir, exist_ok=True)
  log_file_path = os.path.join(log_dir, f'{formatted_time}.log')

  formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

  file_handler = logging.FileHandler(log_file_path)
  file_handler.setFormatter(formatter)

  console_handler = logging.StreamHandler(sys.stdout)
  console_handler.setFormatter(formatter)

  logger.addHandler(file_handler)
  logger.addHandler(console_handler)

  return logger