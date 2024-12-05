from torch.utils.data import Dataset

class DTIDataset(Dataset):
    def __init__(self, protein_ontological, protein_similarity, protein_llm_embeddings, 
                 drug_ontological, drug_similarity, drug_llm_embeddings, 
                 protein_drug_matrix, protein_order, drug_order):
        self.protein_ontological = protein_ontological
        self.protein_similarity = protein_similarity
        self.protein_llm_embeddings = protein_llm_embeddings
        
        self.drug_ontological = drug_ontological
        self.drug_similarity = drug_similarity
        self.drug_llm_embeddings = drug_llm_embeddings
        
        self.protein_drug_matrix = protein_drug_matrix
        self.protein_order = protein_order
        self.drug_order = drug_order

    def __len__(self):
        return len(self.protein_order) * len(self.drug_order)
    
    def __getitem__(self, idx):
        # 根据 idx 计算蛋白质和药物的索引
        protein_idx = idx // len(self.drug_order)
        drug_idx = idx % len(self.drug_order)
        
        
        protein_ontological_feature = self.protein_ontological[protein_idx]
        protein_similarity_feature = self.protein_similarity[protein_idx]
        protein_llm_embedding = self.protein_llm_embeddings[protein_idx]
        
        drug_ontological_feature = self.drug_ontological[drug_idx]
        drug_similarity_feature = self.drug_similarity[drug_idx]
        drug_llm_embedding = self.drug_llm_embeddings[drug_idx]
        
        # 获取蛋白质和药物的结合关系（标签）
        label = self.protein_drug_matrix[protein_idx, drug_idx]
        
        return (protein_ontological_feature, drug_ontological_feature, 
                protein_similarity_feature, drug_similarity_feature,
                protein_llm_embedding, drug_llm_embedding, label)
