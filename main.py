import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, roc_curve, \
    precision_recall_curve, auc
import matplotlib.pyplot as plt

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
# 定义 MLP 分类器
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(0.4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# 加载数据
data = pd.read_csv("../data-c/data-0.3.csv")
features = data.iloc[:, :-1].values
labels = data.iloc[:, -1].values


# 定义函数进行交叉验证并计算平均AUC、AUPR及其他指标
def cross_validation_eval(features, labels, n_splits, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    auc_scores, aupr_scores = [], []
    accuracy_scores, f1_scores, precision_scores, recall_scores = [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    tprs, precisions = [], []
    best_model_state = None  # 用于保存最佳模型的状态字典
    best_auc = 0  # 用于记录最高 AUC

    for train_index, test_index in kf.split(features):
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

        # 定义和训练模型
        input_dim = X_train.shape[1]
        model = MLPClassifier(input_dim, hidden_dim=64, output_dim=2)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)

        # 训练模型
        best_loss = float('inf')
        patience = 20
        patience_counter = 0
        num_epochs = 400
        model.train()
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            l2_lambda = 0.001
            l2_reg = sum(torch.norm(param, 2) for param in model.parameters())
            loss += l2_lambda * l2_reg
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break

        # 模型评估
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            outputs = model(X_test_tensor)
            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = np.argmax(outputs.cpu().numpy(), axis=1)

        # 计算并存储指标
        auc_score = roc_auc_score(y_test, probs)
        fpr, tpr, _ = roc_curve(y_test, probs)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        auc_scores.append(auc_score)

        precision, recall, _ = precision_recall_curve(y_test, probs)
        aupr_score = auc(recall, precision)
        interp_precision = np.interp(mean_recall, recall[::-1], precision[::-1])
        precisions.append(interp_precision)
        aupr_scores.append(aupr_score)

        # 其他指标
        accuracy_scores.append(accuracy_score(y_test, preds))
        f1_scores.append(f1_score(y_test, preds))
        precision_scores.append(precision_score(y_test, preds))
        recall_scores.append(recall_score(y_test, preds))
        # 检查当前模型是否为最佳模型
        if auc_score > best_auc:
            best_auc = auc_score
            best_model_state = model.state_dict()  # 更新最佳模型

        # 保存最佳模型
    torch.save(best_model_state, "../final_5fold_model.pth")
    print("Final 5-fold cross-validation model saved as 'final_5fold_model.pth'")

    # 计算平均AUC、AUPR曲线及其他指标
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)

    mean_precision = np.mean(precisions, axis=0)
    mean_aupr = auc(mean_recall, mean_precision)

    mean_accuracy = np.mean(accuracy_scores)
    mean_f1 = np.mean(f1_scores)
    mean_precision_score = np.mean(precision_scores)
    mean_recall_score = np.mean(recall_scores)

    return mean_fpr, mean_tpr, mean_auc, mean_recall, mean_precision, mean_aupr, mean_accuracy, mean_f1, mean_precision_score, mean_recall_score


# 计算2折、5折和10折的平均曲线和指标
results = {}
for n_splits in [2, 5, 10]:
    mean_fpr, mean_tpr, mean_auc, mean_recall, mean_precision, mean_aupr, mean_accuracy, mean_f1, mean_precision_score, mean_recall_score = cross_validation_eval(
        features, labels, n_splits)
    results[n_splits] = (
    mean_fpr, mean_tpr, mean_auc, mean_recall, mean_precision, mean_aupr, mean_accuracy, mean_f1, mean_precision_score,
    mean_recall_score)

    # 输出平均指标
    print(f"{n_splits}-Fold Cross Validation Results:")
    print(f"Mean AUC: {mean_auc:.4f}")
    print(f"Mean AUPR: {mean_aupr:.4f}")
    print(f"Mean Accuracy: {mean_accuracy:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    print(f"Mean Precision: {mean_precision_score:.4f}")
    print(f"Mean Recall: {mean_recall_score:.4f}")
    print("")

# import matplotlib.pyplot as plt
#
#
#
# colors = ['#007bff', '#28a745', '#fd7e14']  # 选择的颜色
#
# # 绘制 AUC 对比曲线
# plt.figure(figsize=(8, 6))  # 4:3比例
# for i, (n_splits, (mean_fpr, mean_tpr, mean_auc, _, _, _, _, _, _, _)) in enumerate(results.items()):
#     plt.plot(mean_fpr, mean_tpr, lw=2, color=colors[i], label=f'{n_splits}-Fold AUC={mean_auc:.4f}')
#
# plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=2)
# plt.xlabel('False Positive Rate', fontsize=14)
# plt.ylabel('True Positive Rate', fontsize=14)
# plt.title('ROC curve', fontsize=16)
# plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
# plt.gca().set_aspect(3/4)  # 调整曲线为4:3比例
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.tight_layout()
# plt.savefig("data/auc_curve1.png", dpi=300)
# plt.show()
#
# # 绘制放大 AUC 图
# plt.figure(figsize=(8, 6))  # 4:3比例
# for i, (n_splits, (mean_fpr, mean_tpr, mean_auc, _, _, _, _, _, _, _)) in enumerate(results.items()):
#     plt.plot(mean_fpr, mean_tpr, lw=2, color=colors[i], label=f'{n_splits}-Fold AUC={mean_auc:.4f}')
#
# plt.xlabel('False Positive Rate', fontsize=14)
# plt.ylabel('True Positive Rate', fontsize=14)
# plt.title('Enlarged version of partial ROC curve', fontsize=16)
# plt.legend(loc='lower right', frameon=True, fancybox=True, shadow=True, fontsize=12)
# plt.gca().set_aspect(3/4)
# plt.xlim([0.1, 0.2])
# plt.ylim([0.9, 1])
# plt.tight_layout()
# plt.savefig("data/auc_zoomed1.png", dpi=300)
# plt.show()
#
# # 绘制 AUPR 对比曲线
# plt.figure(figsize=(8, 6))  # 4:3比例
# for i, (n_splits, (_, _, _, mean_recall, mean_precision, mean_aupr, _, _, _, _)) in enumerate(results.items()):
#     mean_recall = np.concatenate([mean_recall, [1]])
#     mean_precision = np.concatenate([mean_precision, [0]])
#     plt.plot(mean_recall, mean_precision, lw=2, color=colors[i], label=f'{n_splits}-Fold AUPR={mean_aupr:.4f}')
#
# plt.plot([0, 1], [1, 0], linestyle='--', color='gray', lw=2)
# plt.xlabel('Recall', fontsize=14)
# plt.ylabel('Precision', fontsize=14)
# plt.title('P-R curve', fontsize=16)
# plt.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=12)
# plt.gca().set_aspect(3/4)
# plt.xlim([-0.05, 1.05])
# plt.ylim([-0.05, 1.05])
# plt.tight_layout()
# plt.savefig("data/aupr_curve1.png", dpi=300)
# plt.show()
#
# # 绘制放大 AUPR 图
# plt.figure(figsize=(8, 6))  # 4:3比例
# for i, (n_splits, (_, _, _, mean_recall, mean_precision, mean_aupr, _, _, _, _)) in enumerate(results.items()):
#     plt.plot(mean_recall, mean_precision, lw=2, color=colors[i], label=f'{n_splits}-Fold AUPR={mean_aupr:.4f}')
#
# plt.xlabel('Recall', fontsize=14)
# plt.ylabel('Precision', fontsize=14)
# plt.title('Enlarged version of partial P-R curve', fontsize=16)
# plt.legend(loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=12)
# plt.gca().set_aspect(3/4)
# plt.xlim([0.8, 0.9])
# plt.ylim([0.9, 1])
# plt.tight_layout()
# plt.savefig("data/aupr_zoomed1.png", dpi=300)
# plt.show()