from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import math


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V):
        attention = torch.matmul(Q, torch.transpose(K, -1, -2))
        # 单独把缩放因子拿出来了,控制一下数值类型
        scaling_factor = torch.sqrt(torch.tensor(Q.size(-1), dtype=torch.float32))
        attention = attention / scaling_factor

        attention = torch.softmax(attention, dim=-1)
        output = torch.matmul(attention, V)
        return output


class MultiCrossAttention(nn.Module):
    def __init__(self, hidden_size, all_head_size, head_num):
        super().__init__()
        self.hidden_size = hidden_size
        self.all_head_size = all_head_size
        self.num_heads = head_num
        self.h_size = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V  (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_output = nn.Linear(all_head_size, hidden_size)

        # normalization
        self.norm = sqrt(all_head_size)
        # 一般attention 可以在这里定义
        self.attention = CalculateAttention()

    def forward(self, x, y):
        # x (B, H), y (B, H)
        batch_size = x.size(0)
        # (B, H) -proj-> (B, O) -split-> (B, N, D)

        # q_s: [batch_size, num_heads, h_size]
        q_s = self.linear_q(x).view(batch_size, self.num_heads, self.h_size)

        # k_s: [batch_size, num_heads, h_size]
        k_s = self.linear_k(y).view(batch_size, self.num_heads, self.h_size)

        # v_s: [batch_size, num_heads, h_size]
        v_s = self.linear_v(y).view(batch_size, self.num_heads, self.h_size)

        output = self.attention(q_s, k_s, v_s)

        # attention : [batch_size , num_heads * h_size]
        output = output.transpose(1, 2).contiguous().view(batch_size, self.num_heads * self.h_size)

        # output : [batch_size , hidden_size]
        output = self.linear_output(output)

        return output


class TSFusion(nn.Module):
    def __init__(self, num_hidden_a, num_hidden_b, num_hidden):
        super(TSFusion, self).__init__()
        self.hidden = num_hidden
        self.w1 = nn.Parameter(torch.Tensor(num_hidden_a, num_hidden))
        self.w2 = nn.Parameter(torch.Tensor(num_hidden_b, num_hidden))
        self.bias = nn.Parameter(torch.Tensor(num_hidden))
        self.reset_parameter()

    def reset_parameter(self):
        stdv1 = 1. / math.sqrt(self.hidden)
        stdv2 = 1. / math.sqrt(self.hidden)
        stdv = (stdv1 + stdv2) / 2.
        self.w1.data.uniform_(-stdv1, stdv1)
        self.w2.data.uniform_(-stdv2, stdv2)
        self.bias.data.uniform_(-stdv, stdv)

    def forward(self, a, b):
        wa = torch.matmul(a, self.w1)
        wb = torch.matmul(b, self.w2)
        gated = wa + wb + self.bias
        gate = torch.sigmoid(gated)
        # print(gate.size(),a.size())
        output = gate * a + (1 - gate) * b
        return output  # Clone the tensor to make it out of place operation
    
    
# class TPFModel(nn.Module):
#     def __init__(self, input_dims, hidden_dim):
#         super(TPFModel, self).__init__()
#         self.fusion_moe = TSFusion(input_dims[0], input_dims[0], hidden_dim)
#         self.regressor = nn.Linear(hidden_dim, 1)  # 简单的分类器

#     def forward(self, a, b):
#         fused_rep = self.fusion_moe(a, b)
#         output = self.regressor(fused_rep).squeeze(1)
#         return output


class LLM3Model(nn.Module):
    def __init__(self, cfg):
        super(LLM3Model, self).__init__()
      
        self.drug_stru_1 = nn.Linear(100+708, 1024)
        self.drug_stru_2 = nn.Linear(1024, 256)
        
        self.drug_llm_1 = nn.Linear(4096, 1024)
        self.drug_llm_2 = nn.Linear(1024, 256)

        self.prot_stru_1 = nn.Linear(400+1493, 1024)
        self.prot_stru_2 = nn.Linear(1024, 256)       
        
        self.prot_llm_1 = nn.Linear(4096, 1024)
        self.prot_llm_2 = nn.Linear(1024, 256)   

        
        self.cra = MultiCrossAttention(
            hidden_size=cfg.hidden_size,
            all_head_size=cfg.all_head_size,
            head_num=cfg.head_num
        )
    
        self.fusion = TSFusion(cfg.hidden_size, cfg.hidden_size, cfg.hidden_size)
        # self.prot_fusion = TSFusion(cfg.hidden_size, cfg.hidden_size, cfg.hidden_size)
        
        self.cls_1 = nn.Linear(
            in_features=cfg.hidden_size*2,
            # in_features=100+708+400+1493,
            out_features=cfg.hidden_size
        )
        self.cls_2 = nn.Linear(
            in_features=cfg.hidden_size,
            out_features=cfg.label_num
        )

        self.relu = nn.ReLU()
        self.drop_out_1 = nn.Dropout(0.3)
        self.drop_out_2 = nn.Dropout(0.1)
        
        self.initialize_weights()
       
       
    def initialize_weights(self):
        """自定义参数初始化方法"""
        for m in self.modules():
            # （1）处理线性层
            if isinstance(m, nn.Linear):
                # 对于 ReLU 激活后的线性层，用 Kaiming 初始化（何凯明初始化）
                if hasattr(self, 'relu'):  # 模型中使用了 ReLU
                    init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                else:
                    # 其他情况用 Xavier 初始化
                    init.xavier_normal_(m.weight)
                # 偏置初始化为 0（若存在偏置）
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            
            # （2）处理 MultiCrossAttention 中的线性层（确保多头一致性）
            elif isinstance(m, MultiCrossAttention):
                # 对 Q/K/V 投影矩阵进行 Xavier 初始化（注意力层更关注分布一致性）
                for name, param in m.named_parameters():
                    if 'linear_q.weight' in name or 'linear_k.weight' in name or 'linear_v.weight' in name:
                        init.xavier_normal_(param)
                    if 'linear_output.weight' in name:
                        init.xavier_normal_(param)
                    # 偏置初始化为 0
                    if 'bias' in name and param is not None:
                        init.constant_(param, 0.0)
            
            # （3）处理 TSFusion 中的自定义参数（w1, w2, bias）
            # elif isinstance(m, TSFusion):
            #     # 重新初始化 w1 和 w2（覆盖原 reset_parameter 方法，更适配 ReLU）
            #     init.kaiming_uniform_(m.w1, mode='fan_in', nonlinearity='relu')
            #     init.kaiming_uniform_(m.w2, mode='fan_in', nonlinearity='relu')
            #     init.constant_(m.bias, 0.0)  # 偏置初始化为 0 

    def forward(self, features):
        # (B, E)
        drug_structured, drug_llm_embeddings, prot_structured, protein_llm_embeddings = features
        # (B, 1, E)
        
        # (B, O, *)
        drug_structured = self.drop_out_1(self.relu(self.drug_stru_1(drug_structured)))
        drug_structured = self.drop_out_2(self.relu(self.drug_stru_2(drug_structured)))
        
        drug_llm_embeddings = self.drop_out_1(self.relu(self.drug_llm_1(drug_llm_embeddings)))
        drug_llm_embeddings = self.drop_out_2(self.relu(self.drug_llm_2(drug_llm_embeddings)))
        
        prot_structured = self.drop_out_1(self.relu(self.prot_stru_1(prot_structured)))
        prot_structured = self.drop_out_2(self.relu(self.prot_stru_2(prot_structured)))
        
        protein_llm_embeddings = self.drop_out_1(self.relu(self.prot_llm_1(protein_llm_embeddings)))
        protein_llm_embeddings = self.drop_out_2(self.relu(self.prot_llm_2(protein_llm_embeddings)))
        
        """ablation cra"""
        # drug_structured = self.cra(drug_structured, drug_structured)
        # prot_structured = self.cra(prot_structured, prot_structured)
        
        # drug_llm_embeddings = self.cra(drug_llm_embeddings, drug_llm_embeddings)
        # protein_llm_embeddings = self.cra(protein_llm_embeddings, protein_llm_embeddings)  
        
        drug_structured = self.cra(drug_structured, drug_llm_embeddings)
        prot_structured = self.cra(prot_structured, protein_llm_embeddings)
        
        drug_llm_embeddings = self.cra(drug_llm_embeddings, drug_structured)
        protein_llm_embeddings = self.cra(protein_llm_embeddings, prot_structured)    
        
        """ablation gated"""
        # drug_features = drug_structured + drug_llm_embeddings
        # prot_features = prot_structured + protein_llm_embeddings
        
        drug_features = self.fusion(drug_structured, drug_llm_embeddings)
        prot_features = self.fusion(prot_structured, protein_llm_embeddings)
        
        dt_features = torch.cat((drug_features, prot_features), dim=-1)

        # decode
        pred = self.relu(self.cls_1(dt_features))
        pred = self.cls_2(pred)
        # pred = self.cls_3(pred)

        return dt_features, pred