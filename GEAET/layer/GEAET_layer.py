import numpy as np
import torch
import torch.nn as nn
import torch_geometric.graphgym.register as register
import torch_geometric.nn as pygnn
from performer_pytorch import SelfAttention
from torch_geometric.data import Batch
from torch_geometric.nn import Linear as Linear_pyg
from torch_geometric.utils import to_dense_batch

from GEAET.layer.bigbird_layer import SingleBigBirdLayer
from GEAET.layer.gatedgcn_layer import GatedGCNLayer
from GEAET.layer.gine_conv_layer import GINEConvESLapPE
from GEAET.layer.external_layer import GEANet


class GEAETLayer(nn.Module):
    """Local MPNN + full graph attention x-former layer.
    """

    def __init__(self, dim_h,
                 local_gnn_type, global_model_type,  external_model_type,num_heads, act='relu',
                 pna_degrees=None, equivstable_pe=False, ffn_dropout=0.0,
                 global_dropout=0.0,local_dropout=0.0,external_dropout=0.0,
                 attn_dropout=0.0, layer_norm=False, batch_norm=True,
                 bigbird_cfg=None, log_attn_weights=False,GEANet_cfg=None,use_ffn=True,local_out_act=False):
        super().__init__()

        self.dim_h = dim_h
        self.num_heads = num_heads
        self.attn_dropout = attn_dropout
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.equivstable_pe = equivstable_pe
        self.activation = register.act_dict[act]
        self.use_ffn = use_ffn
        self.use_local_out_act = local_out_act

        self.log_attn_weights = log_attn_weights
        if log_attn_weights and global_model_type not in ['Transformer',
                                                          'BiasedTransformer']:
            raise NotImplementedError(
                f"Logging of attention weights is not supported "
                f"for '{global_model_type}' global attention model."
            )

        # Local message-passing model.
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None

        # MPNNs without edge attributes support.
        elif local_gnn_type == "GCN":
            self.local_gnn_with_edge_attr = False
            self.local_model = pygnn.GCNConv(dim_h, dim_h)
        elif local_gnn_type == 'GIN':
            self.local_gnn_with_edge_attr = False
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            self.local_model = pygnn.GINConv(gin_nn)

        # MPNNs supporting also edge attributes.
        elif local_gnn_type == 'GENConv':
            self.local_model = pygnn.GENConv(dim_h, dim_h)
        elif local_gnn_type == 'GINE':
            gin_nn = nn.Sequential(Linear_pyg(dim_h, dim_h),
                                   self.activation(),
                                   Linear_pyg(dim_h, dim_h))
            if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
                self.local_model = GINEConvESLapPE(gin_nn)
            else:
                self.local_model = pygnn.GINEConv(gin_nn)
        elif local_gnn_type == 'GAT':
            self.local_model = pygnn.GATConv(in_channels=dim_h,
                                             out_channels=dim_h // num_heads,
                                             heads=num_heads,
                                             edge_dim=dim_h)
        elif local_gnn_type == 'PNA':
            # Defaults from the paper.
            # aggregators = ['mean', 'min', 'max', 'std']
            # scalers = ['identity', 'amplification', 'attenuation']
            aggregators = ['mean', 'max', 'sum']
            scalers = ['identity']
            deg = torch.from_numpy(np.array(pna_degrees))
            self.local_model = pygnn.PNAConv(dim_h, dim_h,
                                             aggregators=aggregators,
                                             scalers=scalers,
                                             deg=deg,
                                             edge_dim=min(128, dim_h),
                                             towers=1,
                                             pre_layers=1,
                                             post_layers=1,
                                             divide_input=False)
        elif local_gnn_type == 'CustomGatedGCN':
            self.local_model = GatedGCNLayer(dim_h, dim_h,
                                             dropout=ffn_dropout,
                                             residual=True,
                                             act=act,
                                             equivstable_pe=equivstable_pe)
        else:
            raise ValueError(f"Unsupported local GNN model: {local_gnn_type}")
        self.local_gnn_type = local_gnn_type

        if self.use_local_out_act:
            self.local_out_act = register.act_dict[act]()

        # Global attention transformer-style model.
        if global_model_type == 'None':
            self.global_model = None
        elif global_model_type in ['Transformer', 'BiasedTransformer']:
            self.global_model = torch.nn.MultiheadAttention(
                dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
            # self.global_model = torch.nn.TransformerEncoderLayer(
            #     d_model=dim_h, nhead=num_heads,
            #     dim_feedforward=2048, dropout=0.1, activation=F.relu,
            #     layer_norm_eps=1e-5, batch_first=True)
        elif global_model_type == 'Performer':
            self.global_model = SelfAttention(
                dim=dim_h, heads=num_heads,
                dropout=self.attn_dropout, causal=False)
        # elif global_model_type == "BigBird":
        #     bigbird_cfg.dim_hidden = dim_h
        #     bigbird_cfg.n_heads = num_heads
        #     bigbird_cfg.dropout = ffn_dropout
        #     self.global_model = SingleBigBirdLayer(bigbird_cfg)
        else:
            raise ValueError(f"Unsupported global x-former model: "
                             f"{global_model_type}")
        self.global_model_type = global_model_type


        # External attention model.
        if external_model_type == 'None':
            self.external_model = None
        elif external_model_type == 'GEANet':
            self.external_model = GEANet(dim_h, GEANet_cfg)

        self.external_model_type = external_model_type


        if self.layer_norm and self.batch_norm:
            raise ValueError("Cannot apply two types of normalization together")

        # Normalization for MPNN and Self-Attention representations.
        if self.layer_norm:
            if self.local_model is not None:
                self.norm1_local = pygnn.norm.LayerNorm(dim_h)
            if self.global_model is not None:
                self.norm1_global = pygnn.norm.LayerNorm(dim_h)
            if self.external_model is not None:
                self.norm1_external = pygnn.norm.LayerNorm(dim_h)
            # self.norm1_local = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_attn = pygnn.norm.GraphNorm(dim_h)
            # self.norm1_local = pygnn.norm.InstanceNorm(dim_h)
            # self.norm1_attn = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            if self.local_model is not None:
                self.norm1_local = nn.BatchNorm1d(dim_h)
            if self.global_model is not None:
                self.norm1_global = nn.BatchNorm1d(dim_h)
            if self.external_model is not None:
                self.norm1_external = nn.BatchNorm1d(dim_h)

        if self.local_model is not None:
            self.dropout_local = nn.Dropout(local_dropout)
        if self.global_model is not None:
            self.dropout_global = nn.Dropout(global_dropout)
        if self.external_model is not None:
            self.dropout_external = nn.Dropout(external_dropout)

        if self.use_ffn:
            # Feed Forward block.
            self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
            self.act_fn_ff = self.activation()
            
            self.ff_dropout1 = nn.Dropout(ffn_dropout)
            self.ff_dropout2 = nn.Dropout(ffn_dropout)

        if self.layer_norm:
            self.norm2 = pygnn.norm.LayerNorm(dim_h)
                # self.norm2 = pygnn.norm.GraphNorm(dim_h)
                # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        if self.batch_norm:
            self.norm2 = nn.BatchNorm1d(dim_h)

    def forward(self, batch):
        h = batch.x
        h_in1 = h  # for first residual connection

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_model is not None:
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.
            if self.local_gnn_type == 'CustomGatedGCN':
                es_data = None
                if self.equivstable_pe:
                    es_data = batch.pe_EquivStableLapPE
                local_out = self.local_model(Batch(batch=batch,
                                                   x=h,
                                                   edge_index=batch.edge_index,
                                                   edge_attr=batch.edge_attr,
                                                   pe_EquivStableLapPE=es_data))
                # GatedGCN does residual connection and dropout internally.
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            else:
                if self.local_gnn_with_edge_attr:
                    if self.equivstable_pe:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr,
                                                   batch.pe_EquivStableLapPE)
                    else:
                        h_local = self.local_model(h,
                                                   batch.edge_index,
                                                   batch.edge_attr)
                else:
                    h_local = self.local_model(h, batch.edge_index)
                    
                if self.use_local_out_act:
                    h_local = self.local_out_act(h_local)
                    
                h_local = self.dropout_local(h_local)
                h_local = h_in1 + h_local  # Residual connection.

            if self.layer_norm:
                h_local = self.norm1_local(h_local, batch.batch)
            if self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list.append(h_local)

        # Multi-head attention.
        if self.global_model is not None:
            h_dense, mask = to_dense_batch(h, batch.batch)
            if self.global_model_type == 'Transformer':
                h_global = self._sa_block(h_dense, None, ~mask)[mask]
            elif self.global_model_type == 'BiasedTransformer':
                # Use Graphormer-like conditioning, requires `batch.attn_bias`.
                h_global = self._sa_block(h_dense, batch.attn_bias, ~mask)[mask]
            elif self.global_model_type == 'Performer':
                h_global = self.global_model(h_dense, mask=mask)[mask]
            elif self.global_model_type == 'BigBird':
                h_global = self.global_model(h_dense, attention_mask=mask)
            else:
                raise RuntimeError(f"Unexpected {self.global_model_type}")

            h_global = self.dropout_global(h_global)
            h_global = h_in1 + h_global  # Residual connection.
            if self.layer_norm:
                h_global = self.norm1_global(h_global, batch.batch)
            if self.batch_norm:
                h_global = self.norm1_global(h_global)
            h_out_list.append(h_global)

        if self.external_model is not None:
            if self.external_model_type == 'GEANet':
                h_external,batch.edge_attr = self.external_model(h,batch.edge_attr) 
                # h_external = self.dropout_external(h_external)
                # h_external = h_in1 + h_external
            else:
                raise RuntimeError(f"Unexpected {self.external_model_type}")
            h_external = self.dropout_external(h_external)
            h_external = h_in1 + h_external  # Residual connection.
            if self.layer_norm:
                h_external = self.norm1_external(h_external, batch.batch)
            if self.batch_norm:
                h_external = self.norm1_external(h_external)
            h_out_list.append(h_external)
            
        # Combine local and global outputs.
        # h = torch.cat(h_out_list, dim=-1)
        h = sum(h_out_list)

        if self.use_ffn:
            # Feed Forward block.
            h = h + self._ff_block(h)
        if self.layer_norm:
            h = self.norm2(h, batch.batch)
        if self.batch_norm:
            h = self.norm2(h)

        batch.x = h
        return batch

    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block.
        """
        if not self.log_attn_weights:
            x = self.global_model(x, x, x,
                               attn_mask=attn_mask,
                               key_padding_mask=key_padding_mask,
                               need_weights=False)[0]
        else:
            # Requires PyTorch v1.11+ to support `average_attn_weights=False`
            # option to return attention weights of individual heads.
            x, A = self.global_model(x, x, x,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask,
                                  need_weights=True,
                                  average_attn_weights=False)
            self.attn_weights = A.detach().cpu()
        return x

    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    def extra_repr(self):
        s = f'summary: dim_h={self.dim_h}, ' \
            f'local_gnn_type={self.local_gnn_type}, ' \
            f'global_model_type={self.global_model_type}, ' \
            f'heads={self.num_heads}'
        return s
