

from typing import Union
from torch import Tensor
from torch import nn, sum

from torch_geometric.data import Batch, Data
from torch.nn import init


def external_norm(attn):
    softmax = nn.Softmax(dim=0)  # N
    attn = softmax(attn)  # bs,n,S
    attn = attn/sum(attn, dim=2, keepdim=True)  # bs,n,S
    return attn


class DNorm(nn.Module):
    def __init__(
        self,
        dim1=0,dim2=2
    ):
        super().__init__()
        self.dim1=dim1
        self.dim2=dim2
        self.softmax = nn.Softmax(dim=self.dim1)  # N

    def forward(self, attn: Tensor) -> Tensor:
        #softmax = nn.Softmax(dim=0)  # N
        attn = self.softmax(attn)  # bs,n,S
        attn = attn/sum(attn, dim=self.dim2, keepdim=True)  # bs,n,S
        return attn

class GEANet(nn.Module):

    def __init__(
            self, dim, GEANet_cfg):
        super().__init__()

        
        self.dim = dim
        self.external_num_heads = GEANet_cfg.n_heads
        self.use_shared_unit = GEANet_cfg.shared_unit
        self.use_edge_unit = GEANet_cfg.edge_unit
        self.unit_size = GEANet_cfg.unit_size
        
        # self.q_Linear = nn.Linear(in_dim, gconv_dim - dim_pe)
        self.node_U1 = nn.Linear(self.unit_size, self.unit_size)
        self.node_U2 = nn.Linear(self.unit_size, self.unit_size)
        
        assert self.unit_size * self.external_num_heads == self.dim, "dim must be divisible by external_num_heads"

        # nn.init.xavier_normal_(self.node_m1.weight, gain=1)
        # nn.init.xavier_normal_(self.node_m2.weight, gain=1)
        if  self.use_edge_unit:
            self.edge_U1 = nn.Linear(self.unit_size, self.unit_size)
            self.edge_U2 = nn.Linear(self.unit_size, self.unit_size)
            if self.use_shared_unit:
                self.share_U = nn.Linear(dim, dim)

            # nn.init.xavier_normal_(self.edge_m1.weight, gain=1)
            # nn.init.xavier_normal_(self.edge_m2.weight, gain=1)
            # nn.init.xavier_normal_(self.share_m.weight, gain=1)
        self.norm = DNorm()
        
        
       
        #self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)


    def forward(self, node_x,edge_attr = None) -> Tensor:
        if self.use_shared_unit:
            node_x = self.share_U(node_x)
            edge_attr = self.share_U(edge_attr)
        # x : N x 64
        # External attention
        N, d, head = node_x.size()[0], node_x.size()[1], self.external_num_heads
        node_out = node_x.reshape(N, head,-1)  # Q * 4（head）  ：  N x 16 x 4(head)
        #node_out = node_out.transpose(1, 2)  # (N, 16, 4) -> (N, 4, 16)
        node_out = self.node_U1(node_out)
        attn = self.norm(node_out)  # 行列归一化  N x 16 x 4
        node_out = self.node_U2(attn)
        node_out = node_out.reshape(N, -1)
        
        if self.use_edge_unit:
        
            N, d, head = edge_attr.size()[0], edge_attr.size()[1], self.external_num_heads
            edge_out = edge_attr.reshape(N, -1, head)  # Q * 4（head）  ：  N x 16 x 4(head)
            edge_out = edge_out.transpose(1, 2)  # (N, 16, 4) -> (N, 4, 16)
            edge_out = self.edge_U1(edge_out)
            attn = self.norm(edge_out)  # 行列归一化  N x 16 x 4
            edge_out = self.edge_U2(attn)
            edge_out = edge_out.reshape(N, -1)
        else:
            edge_out = edge_attr
        
        return node_out,edge_out

      