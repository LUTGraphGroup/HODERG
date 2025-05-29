import math
import torch.nn.functional as F
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch.nn import Parameter
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from utils import glorot
from utils import HybridODE
import torch as th
from torch import nn
from dgl import function as fn
from dgl.utils import expand_as_pair
from torch_geometric.nn import GCNConv


class ResidualGINConv(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, apply_func=None, aggregator_type="sum",
                 alpha: float = 0.1, theta: float = None, layer: int = None,
                 init_eps: float = 0.0, learn_eps: bool = False,
                 shared_weights: bool = True, activation=None, **kwargs):

        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        # GIN原有参数
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        self.activation = activation
        self.learn_eps = learn_eps

        # 残差相关新参数
        self.alpha = alpha
        self.beta = 1.0
        if theta is not None and layer is not None:
            self.beta = torch.log(torch.tensor(theta / layer + 1))

        # 可学习epsilon参数
        if learn_eps:
            self.eps = Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

        # 残差权重矩阵
        self.shared_weights = shared_weights
        self.weight1 = Parameter(torch.Tensor(64, 64))  # 假设输出维度64
        if shared_weights:
            self.register_parameter('weight2', None)
        else:
            self.weight2 = Parameter(torch.Tensor(64, 64))

        self.reset_parameters()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def reset_parameters(self):
        glorot(self.weight1)
        if not self.shared_weights:
            glorot(self.weight2)
        if self.learn_eps:
            torch.nn.init.constant_(self.eps, 0.0)
        if self.apply_func is not None:
            self.apply_func.reset_parameters()

    def forward(self, graph, feat, x_0: Tensor, edge_weight=None):
        # 原始GIN计算流程
        _reducer = getattr(fn, self._aggregator_type)
        with graph.local_scope():
            aggregate_fn = fn.copy_u("h", "m")
            if edge_weight is not None:
                graph.edata["_edge_weight"] = edge_weight
                aggregate_fn = fn.u_mul_e("h", "_edge_weight", "m")

            feat_src, feat_dst = expand_as_pair(feat, graph)

            # 确保特征在正确设备上
            feat_src = feat_src.to(self.device).float()
            feat_dst = feat_dst.to(self.device).float()

            graph.srcdata["h"] = feat_src
            graph.update_all(aggregate_fn, _reducer("m", "neigh"))

            # GIN核心公式
            rst = (1 + self.eps) * feat_dst + graph.dstdata["neigh"]

            # 应用MLP（如果存在apply_func）
            if self.apply_func is not None:
                rst = self.apply_func(rst)

            # 残差连接部分
            rst = rst * (1 - self.alpha) + self.alpha * x_0[:rst.size(0)]

            # 权重矩阵混合
            if self.weight2 is None:
                out = torch.addmm(rst, rst, self.weight1,
                                  beta=1. - self.beta, alpha=self.beta)
            else:
                out = torch.addmm(rst, rst, self.weight1,
                                  beta=1. - self.beta, alpha=self.beta)
                out += torch.addmm(x_0, x_0, self.weight2,
                                   beta=1. - self.beta, alpha=self.beta)

            if self.activation is not None:
                out = self.activation(out)

            return out

    def __repr__(self):
        return (f'{self.__class__.__name__}(alpha={self.alpha}, '
                f'beta={self.beta}, eps={self.eps.item()})')


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, dropout=0.5):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList()

        # 输入层 → 隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dim))

        # 中间隐藏层（如果 num_layers > 2）
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # 隐藏层 → 输出层
        self.layers.append(nn.Linear(hidden_dim, output_dim))

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # 或 nn.LeakyReLU(), nn.ELU()

        # 初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)  # Xavier/Glorot 初始化
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # 偏置初始化为 0

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)  # 最后一层不加激活函数
        return x


def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)


def gelu(x):
    """
    GELU activation
    https://arxiv.org/abs/1606.08415
    https://github.com/huggingface/pytorch-openai-transformer-lm/blob/master/model_pytorch.py#L14
    https://github.com/huggingface/pytorch-pretrained-BERT/blob/master/modeling.py
    """
    return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))

# Feedforward Network (FFN)
class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        x = self.layer2(x)
        return x

class BiasedMHA(nn.Module):

    def __init__(
            self,
            feat_size,
            num_heads,
            bias=True,
            attn_bias_type="add",
            attn_drop=0.1,
    ):
        super().__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads
        assert (
                self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5
        self.attn_bias_type = attn_bias_type

        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.u_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.GLU = nn.Linear(feat_size, feat_size, bias=bias)

        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        self.dropout = nn.Dropout(p=attn_drop)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.u_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.GLU.weight, gain=2 ** -0.5)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, ndata, attn_bias=None, attn_mask=None):

        q_h = self.q_proj(ndata).transpose(0, 1)
        k_h = self.k_proj(ndata).transpose(0, 1)
        v_h = self.v_proj(ndata).transpose(0, 1)
        # u_h = self.u_proj(ndata)
        u_h = ndata
        u_h = u_h * torch.sigmoid(self.GLU(u_h))
        bsz, N, _ = ndata.shape
        q_h = (
                q_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(0, 1)
                * self.scaling
        )
        k_h = k_h.reshape(N, bsz * self.num_heads, self.head_dim).permute(
            1, 2, 0
        )
        v_h = v_h.reshape(N, bsz * self.num_heads, self.head_dim).transpose(
            0, 1
        )

        attn_weights = (
            th.bmm(q_h, k_h)
                .transpose(0, 2)
                .reshape(N, N, bsz, self.num_heads)
                .transpose(0, 2)
        )

        if attn_bias is not None:
            if self.attn_bias_type == "add":
                attn_weights += attn_bias
            else:
                attn_weights *= attn_bias
        if attn_mask is not None:
            attn_weights[attn_mask.to(th.bool)] = float("-inf")
        attn_weights = F.softmax(
            attn_weights.transpose(0, 2)
                .reshape(N, N, bsz * self.num_heads)
                .transpose(0, 2),
            dim=2,
        )

        attn_weights = self.dropout(attn_weights)

        attn = th.bmm(attn_weights, v_h).transpose(0, 1)

        attn = self.out_proj(
            attn.reshape(N, bsz, self.feat_size).transpose(0, 1)
        )
        attn = u_h + attn
        return attn

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate,num_heads,attn_bias_type="add",attn_dropout=0.1,):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)


        self.self_attention = BiasedMHA(
            feat_size=hidden_size,
            num_heads=num_heads,
            attn_bias_type=attn_bias_type,
            attn_drop=attn_dropout,
        )
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x, attn_bias=None, attn_mask=None):
        y = self.self_attention_norm(x)
        y = self.self_attention(y, attn_bias, attn_mask)
        y = self.self_attention_dropout(y)
        x = x + y

        y = self.ffn_norm(x)
        y = self.ffn(y)
        y = self.ffn_dropout(y)
        x = x + y
        return x


class InnerProductDecoder(nn.Module):
    """
    decoder 解码器
    """
    def __init__(self, output_node_dim, dropout, num_dis):
        super(InnerProductDecoder, self).__init__()
        self.output_node_dim = output_node_dim
        self.dropout = dropout
        self.num_dis = num_dis
        self.weight = nn.Parameter(torch.empty(size=(self.output_node_dim, self.output_node_dim)))
        nn.init.xavier_uniform_(self.weight.data, gain=1.414)

    def forward(self, inputs):
        inputs = F.dropout(inputs, self.dropout)
        Dis = inputs[0:self.num_dis, :]
        Meta = inputs[self.num_dis:, :]
        Meta = torch.mm(Meta, self.weight)
        Dis = torch.t(Dis)
        x = torch.mm(Meta, Dis)
        outputs = torch.sigmoid(x)
        return outputs


class HODERG(nn.Module):
    def __init__(
            self,
            hops,
            output_dim,
            input_dim,
            pe_dim,
            num_dis,
            num_mi,
            graphformer_layers,
            num_heads,
            hidden_dim,
            ffn_dim,
            dropout_rate,
            GCNII_layers,

    ):

        super().__init__()
        self.seq_len = hops + 1
        self.pe_dim = pe_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads

        self.ode_func = HybridODE(ode_hidden_dim=input_dim)  # 使用input_dim作为ODE输入维度

        self.graphformer_layers = graphformer_layers

        self.dropout_rate = dropout_rate
        self.dropout = nn.Dropout(self.dropout_rate)

        self.num_dis = num_dis
        self.num_mi = num_mi
        self.GCNII_layers = GCNII_layers
        self.convs = nn.ModuleList()


        # Residual Graph Convolution (RGC)
        for i in range(self.GCNII_layers):
            mlp1 = MLP(input_dim=64, hidden_dim=64, output_dim=64, num_layers=2)
            conv = ResidualGINConv(
                  apply_func=mlp1,  # 自定义的MLP
                  alpha=0.1,            # 残差混合比例
                  theta=1,            # beta衰减系数
                  layer=2,              # 当前层数
                  shared_weights=True,# 共享权重矩阵
                  activation=nn.ReLU()
            )

            self.convs.append(conv)

        # Graph Transformer (GT)
        self.att_embeddings_nope = nn.Linear(self.input_dim, self.hidden_dim)
        encoders = [
            EncoderLayer(self.hidden_dim, self.ffn_dim, self.dropout_rate, self.num_heads)
            for _ in range(self.graphformer_layers)]
        self.layers = nn.ModuleList(encoders)
        self.final_ln = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(self.hidden_dim, int(self.hidden_dim / 2))
        self.attn_layer = nn.Linear(2 * self.hidden_dim, 1)
        self.Linear1 = nn.Linear(int(self.hidden_dim / 2), self.output_dim)
        self.scaling = nn.Parameter(torch.ones(1) * 0.5)

        # Multi-layer perceptron (MLP)
        self.mlp = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout_rate),
            nn.Linear(256, 64),
        )
        self.decoder = InnerProductDecoder(self.output_dim, self.dropout_rate, self.num_dis)
        self.apply(lambda module: init_params(module, n_layers=self.graphformer_layers))

    def get_ode_params(self):
        """获取ODE的可学习参数（alpha和beta）"""
        return [self.ode_func.alpha, self.ode_func.beta]



    def forward(self,ode_features,dis_data, mi_data,dis_data1,mi_data1,g,g1):
        # Residual graph convolution for disease similarity network coding
        x_0_dis = dis_data1.x
        x_0_dis1=x_0_dis.clone()
        for conv in self.convs:
            x_dis = conv(g,x_0_dis,x_0=x_0_dis1)


        x_0_mi = mi_data1.x
        x_0_mi1=x_0_mi.clone()

        for conv in self.convs:
            x_mi = conv(g1,x_0_mi,x_0=x_0_mi1)

        x_GCNII = torch.cat((x_dis, x_mi), dim=0)


        tensor = self.att_embeddings_nope(ode_features)


        # transformer encoder
        for enc_layer in self.layers:
            tensor = enc_layer(tensor)
        x_former = self.final_ln(tensor)

        x_former1 = x_former[:, 0, :]  # 如果需要移除第二个维度


        # # Equation (28) and (29) in the paper
        output = torch.cat((x_GCNII, x_former1), dim=1)
        embedings = self.mlp(output)
        x1 = self.decoder(embedings)
        return x1         #,x_GCNII, x_former1

