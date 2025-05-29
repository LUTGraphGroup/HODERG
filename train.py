from __future__ import division
from __future__ import print_function
import time
import argparse
from model import *
from metric import *
from sklearn import metrics
import torch.nn.functional as F
import dgl
from torch import optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

parser = argparse.ArgumentParser()
# training parameters
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500,
                    help='Number of epochs to train.')
parser.add_argument('--weight_decay', type=float, default=5e-4,
                    help='Weight decay (L2 loss on parameters).')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='Dropout rate (1 - keep probability).')
parser.add_argument('--tot_updates', type=int, default=1000,
                    help='used for optimizer learning rate scheduling')
parser.add_argument('--warmup_updates', type=int, default=400,
                    help='warmup steps')
parser.add_argument('--peak_lr', type=float, default=0.001,
                    help='Initial learning rate')
parser.add_argument('--end_lr', type=float, default=0.0001,
                    help='Final learning rate')




# ODE专用参数
parser.add_argument('--ode_alpha_init', type=float, default=0.5, help='ODE混合系数初始值')
parser.add_argument('--ode_beta_init', type=float, default=1.0, help='ODE温度系数初始值')
parser.add_argument('--ode_time_span', type=float, default=0.05, help='ODE扩散时间跨度')
parser.add_argument('--ode_lr_scale', type=float, default=10.0, help='ODE参数学习率倍数')
# model parameters

parser.add_argument('--pe_dim', type=int, default=15,
                    help='position embedding size')
parser.add_argument('--hops', type=int, default=1,
                    help='Hop of K to be calculated')
parser.add_argument('--graphformer_layers', type=int, default=1,
                    help='number of Graphormer layers')
parser.add_argument('--n_heads', type=int, default=8,
                    help='number of attention heads in Rgcgt.')
parser.add_argument('--node_input', type=int, default=64,
                    help='input dimensions of node features/PCA.')
parser.add_argument('--node_hidden', type=int, default=128,
                    help='hidden dimensions of node features.')
parser.add_argument('--node_output', type=int, default=64,
                    help='output dimensions of node features.')
parser.add_argument('--ffn_dim', type=int, default=256,
                    help='FFN layer size')
parser.add_argument('--GCNII_layers', type=int, default=25,
                    help='number of GCNII layers .')
# Use parse_known_args to ignore unknown args
args, unknown = parser.parse_known_args()
# args = parser.parse_args()print('args', args)

# data read
Adj, M_1,Dis_adj, Mi_adj, feature, random_index, k_folds = load_data(args.seed, args.node_input)

auc_result = []
acc_result = []
pre_result = []
recall_result = []
f1_result = []
prc_result = []
fprs = []
tprs = []
precisions = []
recalls = []
print("seed=%d, evaluating metabolite-disease...." % args.seed)
for k in range(k_folds):  # Five-fold cross validation
    print("------this is %dth cross validation------" % (k + 1))
    # Validation set positive and negative samples
    Or_train = np.matrix(Adj, copy=True)
    val_pos_edge_index = np.array(random_index[k]).T
    val_pos_edge_index = torch.tensor(val_pos_edge_index, dtype=torch.long).to(device)
    # Negative sampling of validation set
    val_neg_edge_index = np.mat(np.where(Or_train < 1)).T.tolist()
    random.seed(args.seed)
    random.shuffle(val_neg_edge_index)
    val_neg_edge_index = val_neg_edge_index[:val_pos_edge_index.shape[1]]
    val_neg_edge_index = np.array(val_neg_edge_index).T
    val_neg_edge_index = torch.tensor(val_neg_edge_index, dtype=torch.long).to(device)

    Or_train[tuple(np.array(random_index[k]).T)] = 0
    train_pos_edge_index = np.mat(np.where(Or_train > 0))
    train_pos_edge_index = torch.tensor(train_pos_edge_index, dtype=torch.long).to(device)
    Or_train_matrix = np.matrix(M_1, copy=True)
    Or_train_matrix[tuple(np.array(random_index[k]).T)] = 0
    or_adj = constructNet(torch.tensor(Or_train_matrix)).to(device)


    # Positional encoding constructs a structural matrix
    lpe = laplacian_positional_encoding(or_adj, args.pe_dim, device)  # args.pe_dim: Positional encoding dimension
    features = torch.cat((feature, lpe), dim=1)  # Equation (16)


    # Construct disease similarity network
    Dis_network = torch.nonzero(Dis_adj, as_tuple=True)
    Dis_network = torch.stack(Dis_network)
    dis_data = Data(x=feature[:Adj.shape[1], ], edge_index=Dis_network)
    g = dgl.graph((Dis_network[0], Dis_network[1]), num_nodes=Dis_adj.shape[0])
    dis_data1 = Data(x=feature[:g.number_of_nodes(), :], edge_index=Dis_network)

    # Construct metabolite similarity network
    Mi_network = torch.nonzero(Mi_adj, as_tuple=True)
    Mi_network = torch.stack(Mi_network)
    mi_data = Data(x=feature[Adj.shape[1]:, ], edge_index=Mi_network)

    g1= dgl.graph((Mi_network[0], Mi_network[1]), num_nodes=Mi_adj.shape[0])
    mi_data1 = Data(x=feature[:g1.number_of_nodes(), :], edge_index=Mi_network)


    ode_func = HybridODE(ode_hidden_dim=features.shape[1]).to(device)

    model = HODERG(hops=args.hops,
                  output_dim=args.node_output,
                  input_dim=features.shape[1],
                  pe_dim=args.pe_dim,
                  num_dis=Adj.shape[1],
                  num_mi=Adj.shape[0],
                  graphformer_layers=args.graphformer_layers,
                  num_heads=args.n_heads,
                  hidden_dim=args.node_hidden,
                  ffn_dim=args.ffn_dim,
                  dropout_rate=args.dropout,
                  GCNII_layers=args.GCNII_layers,
                  ).to(device)
    # print(model)
    # 验证ODE参数
    print(f"ODE参数: alpha={model.ode_func.alpha.item():.2f}, beta={model.ode_func.beta.item():.2f}")
    print('total params:', sum(p.numel() for p in model.parameters()))
    model.to(device)
    # 更可靠的参数分组方法
    ode_params = []
    other_params = []

    # 使用参数的内存地址来确保唯一性比较
    ode_param_ptrs = {model.ode_func.alpha.data_ptr(), model.ode_func.beta.data_ptr()}

    for name, param in model.named_parameters():
        if param.data_ptr() in ode_param_ptrs:
            ode_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {
            'params': other_params,
            'lr': args.peak_lr,
            'weight_decay': args.weight_decay
        },
        {
            'params': ode_params,
            'lr': args.peak_lr * args.ode_lr_scale,  # 应用学习率倍数
            'weight_decay': 0.0  # 禁用权重衰减
        }
    ])
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.peak_lr, weight_decay=args.weight_decay)
    lr_scheduler = PolynomialDecayLR(
        optimizer,
        warmup_updates=args.warmup_updates,
        tot_updates=args.tot_updates,
        lr=args.peak_lr,
        end_lr=args.end_lr,
        power=1.0
    )
    criterion = F.binary_cross_entropy
    best_epoch = 0
    best_auc = 0
    best_acc = 0
    best_prc = 0
    best_tpr = 0
    best_fpr = 0
    best_recall = 0
    best_precision = 0
    for epoch in range(args.epochs):  # 500 epochs per fold
        start = time.time()

        # Model Training
        model.train()
        optimizer.zero_grad()
        train_neg_edge_index = np.mat(np.where(Or_train_matrix < 1)).T.tolist()
        random.shuffle(train_neg_edge_index)
        train_neg_edge_index = train_neg_edge_index[:train_pos_edge_index.shape[1]]
        train_neg_edge_index = np.array(train_neg_edge_index).T
        train_neg_edge_index = torch.tensor(train_neg_edge_index, dtype=torch.long).to(device)

        # 2. ODE特征计算（新增部分）
        with torch.set_grad_enabled(True):  # 根据需求决定是否计算梯度
            ode_features = compute_ode_features(
                adj=or_adj,  # 假设g是邻接矩阵
                features=features,
                K=args.hops,
                ode_func=model.ode_func,
                time_span=0.05
            )  # [N, K+1, D]





        output = model(ode_features, dis_data, mi_data, dis_data1, mi_data1, g, g1).to(device)


        edge_index = torch.cat([train_pos_edge_index, train_neg_edge_index], 1)
        trian_scores = output[edge_index[0], edge_index[1]].to(device)
        trian_labels = get_link_labels(train_pos_edge_index, train_neg_edge_index).to(device)

        loss_train=criterion(trian_scores, trian_labels).to(device)
        loss_train.backward(retain_graph=True)

        # 梯度裁剪（使用已经分好组的参数）
        torch.nn.utils.clip_grad_norm_(other_params, max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(ode_params, max_norm=0.1)


        optimizer.step()
        lr_scheduler.step()

        # 6. 参数约束
        model.ode_func._clamp_parameters()

        # Model Evaluation
        model.eval()
        with torch.no_grad():
            score_train_cpu = np.squeeze(trian_scores.detach().cpu().numpy())
            label_train_cpu = np.squeeze(trian_labels.detach().cpu().numpy())
            train_auc = metrics.roc_auc_score(label_train_cpu, score_train_cpu)

            predict_y_proba = output.reshape(Adj.shape[0], Adj.shape[1]).to(device)
            score_val, label_val, metric_tmp = cv_model_evaluate(predict_y_proba, val_pos_edge_index,
                                                                 val_neg_edge_index)

            fpr, tpr, thresholds = metrics.roc_curve(label_val, score_val)
            precision, recall, _ = metrics.precision_recall_curve(label_val, score_val)
            val_auc = metrics.auc(fpr, tpr)
            val_prc = metrics.auc(recall, precision)

            end = time.time()
            print('Epoch:', epoch + 1, 'Train Loss: %.4f' % loss_train.item(),
                  'Acc: %.4f' % metric_tmp[0], 'Pre: %.4f' % metric_tmp[1], 'Recall: %.4f' % metric_tmp[2],
                  'F1: %.4f' % metric_tmp[3],
                  'Train AUC: %.4f' % train_auc, 'Val AUC: %.4f' % val_auc, 'Val PRC: %.4f' % val_prc,
                  'Time: %.2f' % (end - start))
            if metric_tmp[
                0] > best_acc and val_auc > best_auc and val_prc > best_prc:  # Save the optimal index for each fold
                metric_tmp_best = metric_tmp
                best_auc = val_auc
                best_prc = val_prc
                best_epoch = epoch + 1
                best_tpr = tpr
                best_fpr = fpr
                best_recall = recall
                best_precision = precision
    print('Fold:', k + 1, 'Best Epoch:', best_epoch, 'Val acc: %.4f' % metric_tmp_best[0],
          'Val Pre: %.4f' % metric_tmp_best[1],
          'Val Recall: %.4f' % metric_tmp_best[2], 'Val F1: %.4f' % metric_tmp_best[3], 'Val AUC: %.4f' % best_auc,
          'Val PRC: %.4f' % best_prc,
          )

    acc_result.append(metric_tmp_best[0])
    pre_result.append(metric_tmp_best[1])
    recall_result.append(metric_tmp_best[2])
    f1_result.append(metric_tmp_best[3])
    auc_result.append(round(best_auc, 4))
    prc_result.append(round(best_prc, 4))

    fprs.append(best_fpr)
    tprs.append(best_tpr)
    recalls.append(best_recall)
    precisions.append(best_precision)

print('## Training Finished !')
print('-----------------------------------------------------------------------------------------------')
print('Acc', acc_result)
print('Pre', pre_result)
print('Recall', recall_result)
print('F1', f1_result)
print('Auc', auc_result)
print('Prc', prc_result)
print('AUC mean: %.4f, variance: %.4f \n' % (np.mean(auc_result), np.std(auc_result)),
      'Accuracy mean: %.4f, variance: %.4f \n' % (np.mean(acc_result), np.std(acc_result)),
      'Precision mean: %.4f, variance: %.4f \n' % (np.mean(pre_result), np.std(pre_result)),
      'Recall mean: %.4f, variance: %.4f \n' % (np.mean(recall_result), np.std(recall_result)),
      'F1-score mean: %.4f, variance: %.4f \n' % (np.mean(f1_result), np.std(f1_result)),
      'PRC mean: %.4f, variance: %.4f \n' % (np.mean(prc_result), np.std(prc_result)))

# 调用方式1：使用默认设置（保存到./result目录）
plot_auc_curves(fprs, tprs, auc_result)
plot_prc_curves(precisions, recalls, prc_result)
