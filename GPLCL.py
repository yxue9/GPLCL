import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import dgl
import random
from torch_geometric.nn import SAGEConv, HeteroConv
from dgl.nn import SAGEConv, HeteroGraphConv
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.ensemble import RandomForestClassifier

from torch_geometric.nn.inits import glorot

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch.optim as optim

import random


def set_random_seed(seed):
    random.seed(seed)  # Python随机数生成器
    np.random.seed(seed)  # NumPy随机数生成器
    torch.manual_seed(seed)  # PyTorch随机数生成器
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # CUDA随机数生成器（如果使用GPU）


class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)  # 添加 Batch Normalization
        self.dropout = nn.Dropout(0.4)  # 添加 Dropout 层
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)  # 使用 Batch Normalization
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 图提示学习模块 GPF_plus
class GPF_plus(nn.Module):
    def __init__(self, in_channels: int, p_num: int, alpha=0.5):
        super(GPF_plus, self).__init__()

        # 初始化提示参数
        self.p_list = nn.Parameter(torch.Tensor(p_num, in_channels))
        self.a = nn.Linear(in_channels, p_num)

        # 可学习的提示权重参数
        self.alpha = nn.Parameter(torch.tensor(alpha, dtype=torch.float32))  # 动态调整alpha
        self.non_linear = nn.Linear(in_channels, in_channels)  # 额外非线性层

        self.reset_parameters()

    def reset_parameters(self):
        # Glorot 初始化
        nn.init.xavier_uniform_(self.p_list)
        nn.init.xavier_uniform_(self.a.weight)
        self.a.bias.data.fill_(0.0)
        nn.init.xavier_uniform_(self.non_linear.weight)
        self.non_linear.bias.data.fill_(0.0)

    def add(self, x: torch.Tensor):
        # 计算输入特征与提示参数的相似性
        score = self.a(x)
        weight = F.softmax(score, dim=1)

        # 生成提示特征
        p = weight.mm(self.p_list)

        # 增加非线性变换
        p_transformed = F.relu(self.non_linear(p))  # 可以尝试其他激活函数

        # 合成特征：加权原特征和提示特征
        return (1 - self.alpha) * x + self.alpha * p_transformed  # 使用动态alpha调整提示特征的比例


# 创建异构图数据
def build_heterograph(met_disease_matrix, metSimi, disSimi):
    num_metabolites = metSimi.shape[0]
    num_diseases = disSimi.shape[0]

    met_met_src, met_met_dst = np.nonzero(metSimi > 0.5)
    dis_dis_src, dis_dis_dst = np.nonzero(disSimi > 0.5)
    met_dis_src, met_dis_dst = np.nonzero(met_disease_matrix)

    data_dict = {
        ('metabolite', 'met_met_edge', 'metabolite'): (met_met_src, met_met_dst),
        ('disease', 'dis_dis_edge', 'disease'): (dis_dis_src, dis_dis_dst),
        ('metabolite', 'met_dis_edge', 'disease'): (met_dis_src, met_dis_dst),
    }

    num_nodes_dict = {'metabolite': num_metabolites, 'disease': num_diseases}
    g = dgl.heterograph(data_dict, num_nodes_dict)
    return g


# 数据加载和特征处理
def load_data():
    met_disease_assoc = pd.read_csv('data/matrix.csv', header=None).values
    met_met_assoc = pd.read_csv('data/all_MS_file.csv', header=None).values
    disease_disease_assoc = pd.read_csv('data/all_DS_file.csv', header=None).values

    g = build_heterograph(met_disease_assoc, met_met_assoc, disease_disease_assoc)

    metabolite_features = torch.tensor(met_met_assoc, dtype=torch.float32)
    disease_features = torch.tensor(disease_disease_assoc, dtype=torch.float32)

    shared_embedding_dim = 256
    shared_embedding_layer_metabolite = nn.Linear(metabolite_features.shape[1], shared_embedding_dim)
    shared_embedding_layer_disease = nn.Linear(disease_features.shape[1], shared_embedding_dim)

    metabolite_features_mapped = shared_embedding_layer_metabolite(metabolite_features)
    disease_features_mapped = shared_embedding_layer_disease(disease_features)

    features = {
        'metabolite': metabolite_features_mapped,
        'disease': disease_features_mapped
    }

    labels = torch.tensor(met_disease_assoc.flatten(), dtype=torch.float32)
    return g, features, labels


# 特征组合和负样本采样
def sample_data(h, labels, pos_neg_ratio=1):
    metabolite_h = h['metabolite']
    disease_h = h['disease']

    num_metabolites = metabolite_h.size(0)
    num_diseases = disease_h.size(0)

    pos_indices = torch.nonzero(labels == 1).squeeze()
    neg_indices = torch.nonzero(labels == 0).squeeze()

    sampled_neg_indices = np.random.choice(neg_indices, size=len(pos_indices) * pos_neg_ratio, replace=False)

    sampled_indices = torch.cat([pos_indices, torch.tensor(sampled_neg_indices, dtype=torch.long)])
    sampled_labels = labels[sampled_indices]

    combined_features = []
    for i in range(num_metabolites):
        for j in range(num_diseases):
            combined_features.append(torch.cat([metabolite_h[i], disease_h[j]], dim=-1))

    combined_features = torch.stack(combined_features)

    return combined_features[sampled_indices], sampled_labels


# HeteroGraphSAGE 模型
class HeteroGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats):
        super(HeteroGraphSAGE, self).__init__()

        self.conv1 = HeteroGraphConv({
            'met_met_edge': SAGEConv(in_feats, hidden_feats, aggregator_type='pool'),
            'dis_dis_edge': SAGEConv(in_feats, hidden_feats, aggregator_type='pool'),
            'met_dis_edge': SAGEConv(in_feats, hidden_feats, aggregator_type='pool')
        }, aggregate='mean')

        self.conv2 = HeteroGraphConv({
            'met_met_edge': SAGEConv(hidden_feats, out_feats, aggregator_type='pool'),
            'dis_dis_edge': SAGEConv(hidden_feats, out_feats, aggregator_type='pool'),
            'met_dis_edge': SAGEConv(hidden_feats, out_feats, aggregator_type='pool')
        }, aggregate='mean')

    def forward(self, g, inputs):
        h = self.conv1(g, inputs)
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        return h


# 图对比学习（Graph Contrastive Learning）模块
import torch
import torch.nn as nn


class GraphCL(nn.Module):
    def __init__(self, hidden_dim, num_heads=4, temperature=0.5):
        super(GraphCL, self).__init__()
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)
        self.temperature = temperature

    def augment_graph(self, x, method):
        # 选择增强方法
        print("1")
        if method == "node_drop":
            mask = torch.rand(x.size(0)) > 0.3  # 随机丢弃10%的节点
            print("2")
            return x[mask]
        elif method == "feature_mask":
            x = x.clone()
            x[torch.rand_like(x) < 0.3] = 0  # 随机将10%的特征设置为0
            print("3")
            return x
        elif method == "add_noise":
            noise = torch.randn_like(x) * 0.3  # 添加标准差为0.1的高斯噪声
            print("4")
            return x + noise
        elif method == "feature_dropout":
            x = x.clone()
            drop_mask = torch.rand(x.size()) < 0.3  # 随机丢弃10%的特征
            x[drop_mask] = 0
            print("5")
            return x
        else:
            return x

    def info_nce_loss(self, z1, z2):
        # 确保输入大小一致
        min_size = min(z1.size(0), z2.size(0))
        z1 = z1[:min_size]  # 裁剪z1
        z2 = z2[:min_size]  # 裁剪z2

        positive_samples = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        negative_samples = torch.exp(torch.mm(z1, z2.T) / self.temperature).sum(dim=-1)
        loss = -torch.log(positive_samples / (negative_samples + 1e-9)).mean()
        return loss

    def forward(self, h):
        random.seed()
        torch.manual_seed(random.randint(1, 10000))  # 使用一个随机的整数作为种子
        # 从输入字典中获取代谢物和疾病特征
        methods = ["node_drop", "feature_mask", "add_noise", "feature_dropout"]  # 可以根据需求添加或修改增强方法

        # 随机选择增强方法
        view1_metabolite = self.augment_graph(h['metabolite'], method=random.choice(methods))
        view1_disease = self.augment_graph(h['disease'], method=random.choice(methods))

        view2_metabolite = self.augment_graph(h['metabolite'], method=random.choice(methods))
        view2_disease = self.augment_graph(h['disease'], method=random.choice(methods))

        # 打印调试信息
        print(f'View1 Metabolite shape: {view1_metabolite.shape}, View1 Disease shape: {view1_disease.shape}')
        print(f'View2 Metabolite shape: {view2_metabolite.shape}, View2 Disease shape: {view2_disease.shape}')

        # 进行投影
        z1_metabolite = self.projection_head(view1_metabolite)
        z1_disease = self.projection_head(view1_disease)

        z2_metabolite = self.projection_head(view2_metabolite)
        z2_disease = self.projection_head(view2_disease)

        # 应用自注意力机制
        z1_metabolite, _ = self.attention(z1_metabolite, z1_metabolite, z1_metabolite)
        z1_disease, _ = self.attention(z1_disease, z1_disease, z1_disease)

        z2_metabolite, _ = self.attention(z2_metabolite, z2_metabolite, z2_metabolite)
        z2_disease, _ = self.attention(z2_disease, z2_disease, z2_disease)

        # 对代谢物和疾病特征分别计算对比损失
        contrastive_loss_metabolite = self.info_nce_loss(z1_metabolite, z2_metabolite)
        contrastive_loss_disease = self.info_nce_loss(z1_disease, z2_disease)

        # 总对比损失可以是两者的平均或加权和
        total_contrastive_loss = (contrastive_loss_metabolite + contrastive_loss_disease) / 2

        return total_contrastive_loss


# 训练和评估流程
def train_model_with_cross_validation():
    set_random_seed(42)
    g, features, labels = load_data()

    # 提取特征，HeteroGraphSAGE
    h = HeteroGraphSAGE(in_feats=256, hidden_feats=256, out_feats=32)(g, features)

    # 图对比学习 - 用于增强特征的相似性
    graph_cl = GraphCL(hidden_dim=32)
    contrastive_loss = graph_cl(h)

    # 图提示学习 GPF_plus
    h_with_gpf = {
        'metabolite': GPF_plus(in_channels=32, p_num=5).add(h['metabolite']),
        'disease': GPF_plus(in_channels=32, p_num=5).add(h['disease'])
    }

    # 进行负采样和特征组合
    combined_features, sampled_labels = sample_data(h_with_gpf, labels)

    # 使用五折交叉验证
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    auc_scores = []
    aupr_scores = []

    for fold, (train_index, test_index) in enumerate(kf.split(combined_features), 1):
        X_train, X_test = combined_features[train_index].detach().numpy(), combined_features[
            test_index].detach().numpy()
        y_train, y_test = sampled_labels[train_index].detach().numpy(), sampled_labels[test_index].detach().numpy()

        # 初始化 MLP 分类器
        input_dim = X_train.shape[1]  # 输入特征维度
        hidden_dim = 64  # 隐藏层维度
        output_dim = 2  # 二分类
        model = MLPClassifier(input_dim, hidden_dim, output_dim)

        # 定义损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        # 转换数据为 PyTorch 张量
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        # 训练模型
        model.train()
        for epoch in range(300):  # 设置训练轮数
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            # 添加 L2 正则化
            l2_lambda = 0.001  # L2 正则化强度
            l2_reg = sum(torch.norm(param, 2) for param in model.parameters())  # 计算 L2 范数
            loss += l2_lambda * l2_reg  # 将 L2 正则化加入损失中

            loss.backward()
            optimizer.step()

        # 测试模型
        model.eval()
        X_test_tensor = torch.FloatTensor(X_test)
        with torch.no_grad():
            test_output = model(X_test_tensor)
            test_output_probs = torch.softmax(test_output, dim=1)[:, 1].numpy()  # 预测概率

        auc = roc_auc_score(y_test, test_output_probs)
        aupr = average_precision_score(y_test, test_output_probs)
        auc_scores.append(auc)
        aupr_scores.append(aupr)

        print(f'Fold {fold} AUC: {auc:.4f}, AUPR: {aupr:.4f}')

    print(f'Mean AUC: {np.mean(auc_scores):.4f}, Mean AUPR: {np.mean(aupr_scores):.4f}')


train_model_with_cross_validation()
