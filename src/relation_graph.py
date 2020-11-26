"""Relational graph modules"""

#numpy imports
import numpy as np

#torch imports
import torch
import torch.nn as nn
from   torch.nn.parameter  import Parameter
import torch.nn.functional as F
import torch.nn.init       as init
from   torch.nn.modules.utils import _pair
from   torch.nn.modules.conv import _ConvNd
from   torch.autograd import Function

#networkx imports
import networkx as nx
from networkx.utils import py_random_state

def graph_to_edge_index(graph, add_self_loops = False):
    directed_graph = graph.to_directed()
    nodes = np.arange(graph.order())
    directed_graph.add_edges_from(zip(nodes, nodes))
    
    return np.array(directed_graph.edges)

def generate_index(graph_type = 'ws_connected', n=16, sparsity=0.5, p=0.2, seed=123):
    degree = int(n * sparsity)

    if graph_type == 'ws':
        graph = nx.watts_strogatz_graph(n=n, k=degree, p=p, seed=seed)
    elif graph_type == 'ws_connected':
        graph = nx.connected_watts_strogatz_graph(n=n, k=degree, p=p, seed=seed)
    elif graph_type == 'ba':
        graph = nx.barabasi_albert_graph(n=n, m=degree // 2, seed=seed)
    elif graph_type == 'grid':
        graph = nx.grid_2d_graph(degree, n // degree)
    elif graph_type == 'cycle':
        graph = nx.cycle_graph(n=n)
    elif graph_type == 'tree':
        graph = nx.random_tree(n=n, seed=seed)
    elif graph_type == 'er':
        graph = nx.erdos_renyi_graph(n, p, seed = seed)
    elif graph_type == 'powerlaw_cluster_graph':
        graph = nx.powerlaw_cluster_graph(n=n, m=degree, p=p, seed = seed)
    else:
        raise NotImplementedError("Unknown graph type!")
    return graph_to_edge_index(graph)

def compute_size(channel, group, seed=1):
    np.random.seed(seed)
    divide = channel // group
    remain = channel % group

    out = np.zeros(group, dtype=int)
    out[:remain] = divide + 1
    out[remain:] = divide
    out = np.random.permutation(out)
    return out

def compute_densemask(in_channels, out_channels, group_num, edge_index):
    repeat_in = compute_size(in_channels, group_num)
    repeat_out = compute_size(out_channels, group_num)
    mask = np.zeros((group_num, group_num))
    mask[edge_index[:, 0], edge_index[:, 1]] = 1
    mask = np.repeat(mask, repeat_out, axis=0)
    mask = np.repeat(mask, repeat_in, axis=1)
    return mask

def get_mask(in_channels, out_channels, group_num,
             graph_type='ba', sparsity=0.5, p=0.2, talk_mode='dense', seed=123):
    assert group_num <= in_channels and group_num <= out_channels
    # high-level graph edge index
    edge_index_high = generate_index(graph_type=graph_type,
                                     n=group_num, sparsity=sparsity, p=p, seed=seed)
    mask_high = compute_densemask(in_channels, out_channels, group_num, edge_index_high)
    return mask_high

############## Linear model

class TalkLinear(nn.Linear):
    '''Relational graph version of Linear. Neurons "talk" according to the graph structure'''

    def __init__(self, in_channels, out_channels, group_num, bias=False,
                 graph_type='ba', sparsity=0.5, p=0.2, talk_mode='dense', seed=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        group_num = min(in_channels, out_channels, group_num)
            
        super(TalkLinear, self).__init__(
            in_channels, out_channels, bias)

        self.mask = get_mask(in_channels, out_channels, group_num,
                             graph_type, sparsity, p, talk_mode, seed)
        nonzero = np.sum(self.mask)
        self.mask = torch.from_numpy(self.mask).float().to(device)

        self.flops_scale = nonzero / (in_channels * out_channels)
        self.params_scale = self.flops_scale
        self.init_scale = torch.sqrt(out_channels / torch.sum(self.mask.cpu(), dim=0, keepdim=True))

    def forward(self, x):
        weight = self.weight * self.mask
        return F.linear(x, weight, self.bias)


class SymLinear(nn.Module):
    '''Linear with symmetric weight matrices'''

    def __init__(self, in_features, out_features, bias=True):
        super(SymLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        weight = self.weight + self.weight.permute(1, 0)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


############## Conv model

class TalkConv2d(_ConvNd):
    '''Relational graph version of Conv2d. Neurons "talk" according to the graph structure'''

    def __init__(self, in_channels, out_channels, group_num, kernel_size, stride=1,
                 padding=0, dilation=1, bias=False, graph_type='ba', agg='sum',
                 sparsity=0.5, p=0.2, talk_mode='dense', seed=None):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        group_num = min(in_channels, out_channels, group_num)
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(TalkConv2d, self).__init__(
            in_channels, out_channels,
            kernel_size, stride, padding, dilation,
            False, _pair(0), 1, bias, 'zeros')

        self.mask = get_mask(in_channels, out_channels, group_num,
                             graph_type, sparsity, p, talk_mode, seed)
        nonzero = np.sum(self.mask)
        self.mask = torch.from_numpy(self.mask[:, :, np.newaxis, np.newaxis]).float().to(device)

        self.init_scale = torch.sqrt(out_channels / torch.sum(self.mask.cpu(), dim=0, keepdim=True))
        self.flops_scale = nonzero / (in_channels * out_channels)
        self.params_scale = self.flops_scale

    def forward(self, input):
        weight = self.weight * self.mask
        return F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, 1)


class SymConv2d(_ConvNd):
    '''Conv2d with symmetric weight matrices'''

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1,
                 bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SymConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)

    def forward(self, input):
        weight = self.weight + self.weight.permute(1, 0, 2, 3)
        if self.padding_mode == 'circular':
            expanded_padding = ((self.padding[1] + 1) // 2, self.padding[1] // 2,
                                (self.padding[0] + 1) // 2, self.padding[0] // 2)
            return F.conv2d(F.pad(input, expanded_padding, mode='circular'),
                            weight, self.bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


########### Other OPs

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SE(nn.Module):
    """Squeeze-and-Excitation (SE) block w/ Swish activation fun."""

    def __init__(self, in_w, se_w, act_fun):
        super(SE, self).__init__()
        self._construct_class(in_w, se_w, act_fun)

    def _construct_class(self, in_w, se_w, act_fun):
        # AvgPool
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # FC, Swish, FC, Sigmoid
        self.f_ex = nn.Sequential(
            nn.Conv2d(in_w, se_w, kernel_size=1, bias=True),
            act_fun(),
            nn.Conv2d(se_w, in_w, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.f_ex(self.avg_pool(x))


class SparseLinear(nn.Linear):
    '''Sparse Linear layer'''

    def __init__(self, group_num, in_scale, out_scale, bias=False,
                 edge_index=None, flops_scale=0.5, params_scale=0.5):
        # mask is used for reset to zero
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        mask_one = np.ones((out_scale, in_scale), dtype=bool)
        mask_zero = np.zeros((out_scale, in_scale), dtype=bool)
        mask_list = [[mask_one for i in range(group_num)] for j in range(group_num)]
        for i in range(edge_index.shape[0]):
            mask_list[edge_index[i, 0]][edge_index[i, 1]] = mask_zero
        self.mask = np.block(mask_list)
        self.edge_index = edge_index
        # todo: update to pytorch 1.2.0, then use bool() dtype
        self.mask = torch.from_numpy(self.mask).byte().to(device)

        self.flops_scale = flops_scale
        self.params_scale = params_scale

        super(SparseLinear, self).__init__(
            group_num * in_scale, group_num * out_scale, bias)

    def forward(self, x):
        weight = self.weight.clone().masked_fill_(self.mask, 0)
        # pdb.set_trace()
        return F.linear(x, weight, self.bias)


class GroupLinear(nn.Module):
    '''Group conv style linear layer'''

    def __init__(self, in_channels, out_channels, bias=False, group_size=1):
        super(GroupLinear, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group_size = group_size
        self.group_num = in_channels // group_size
        self.in_scale = in_channels // self.group_num
        self.out_scale = out_channels // self.group_num
        assert in_channels % self.group_num == 0
        assert out_channels % self.group_num == 0
        assert in_channels % self.group_size == 0
        # Note: agg_fun is always sum

        self.edge_index = np.arange(self.group_num)[:, np.newaxis].repeat(2, axis=1)
        self.edge_num = self.edge_index.shape[0]
        flops_scale = self.edge_num / (self.group_num * self.group_num)
        params_scale = self.edge_num / (self.group_num * self.group_num)

        self.linear = SparseLinear(self.group_num, self.in_scale, self.out_scale, bias,
                                   edge_index=self.edge_index, flops_scale=flops_scale, params_scale=params_scale)

    def forward(self, x):
        x = self.linear(x)
        return x