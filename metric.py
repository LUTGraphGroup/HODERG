from utils import *
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')



#该函数的作用是计算两个向量的相似度，首先对两个向量进行归一化处理，然后计算它们的内积，得到相似度的值。
def sim(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())


#该函数的作用是计算对比损失，首先根据输入的两个向量计算相似度矩阵，然后对相似度矩阵进行指数函数转换。接着，通过计算对角线元素和归一化因子，得到损失值。对角线元素表示同一样本的相似度，归一化因子用于将相似度矩阵进行归一化，从而得到损失值。
def contrastive_loss(h1, h2, tau=0.7):
    sim_matrix = sim(h1, h2)
    f = lambda x: torch.exp(x / tau)
    matrix_t = f(sim_matrix)
    numerator = matrix_t.diag()
    denominator = torch.sum(matrix_t, dim=-1)
    loss = -torch.log(numerator / denominator).mean()
    return loss

# accuracy, precision, recall and f1_score evaluation metrics
def get_metrics(real_score, predict_score):
    sorted_predict_score = np.array(
        sorted(list(set(predict_score.flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(
        sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]

    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)

    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN

    fpr = FP / (FP + TN)
    tpr = TP / (TP + FN)

    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)

    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return [round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1_score, 4)]


def cv_model_evaluate(output, val_pos_edge_index, val_neg_edge_index):
    edge_index = torch.cat([val_pos_edge_index, val_neg_edge_index], 1)
    val_scores = output[edge_index[0], edge_index[1]].to(device)
    val_labels = get_link_labels(val_pos_edge_index, val_pos_edge_index).to(device)
    return val_scores.cpu().numpy(), val_labels.cpu().numpy(), get_metrics(val_labels.cpu().numpy(), val_scores.cpu().numpy())
