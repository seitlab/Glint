import torch
import numpy as np
import dgl.function as fn
import torch.nn.functional as F
from dgl.nn.pytorch.glob import SortPooling

from magnn import *
from layers import *
from typing import List, Tuple, Union



class GraphCrossModule(torch.nn.Module):
    """
    Description
    -----------
    The Graph Cross Module used by Graph Cross Networks.
    This module only contains graph cross layers.

    Parameters
    ----------
    pool_ratios : Union[float, List[float]]
        The pooling ratios (for keeping nodes) for each layer.
        For example, if `pool_ratio=0.8`, 80\% nodes will be preserved.
        If a single float number is given, all pooling layers will have the
        same pooling ratio.
    in_dim : int
        The number of input node feature channels.
    out_dim : int
        The number of output node feature channels.
    hidden_dim : int
        The number of hidden node feature channels.
    cross_weight : float, optional
        The weight parameter used in graph cross layers
        Default: :obj:`1.0`
    fuse_weight : float, optional
        The weight parameter used at the end of GXN for channel fusion.
        Default: :obj:`1.0`
    """
    def __init__(self, pool_ratios:Union[float, List[float]], in_dim:int,
                 out_dim:int, hidden_dim:int, cross_weight:float=1.,
                 fuse_weight:float=1., dist:int=1, num_cross_layers:int=2):
        super(GraphCrossModule, self).__init__()
        if isinstance(pool_ratios, float):
            pool_ratios = (pool_ratios, pool_ratios)
        self.cross_weight = cross_weight
        self.fuse_weight = fuse_weight
        self.num_cross_layers = num_cross_layers

        # build network
        self.start_gcn_scale1 = TAGConvWithDropout(in_dim, hidden_dim)
        self.start_gcn_scale2 = TAGConvWithDropout(hidden_dim, hidden_dim)
        self.end_gcn = TAGConvWithDropout(2 * hidden_dim, out_dim)

        self.index_select_scale1 = IndexSelect(pool_ratios[0], hidden_dim, act="prelu", dist=dist)
        self.index_select_scale2 = IndexSelect(pool_ratios[1], hidden_dim, act="prelu", dist=dist)
        self.start_pool_s12 = GraphPool(hidden_dim)
        self.start_pool_s23 = GraphPool(hidden_dim)
        self.end_unpool_s21 = GraphUnpool(hidden_dim)
        self.end_unpool_s32 = GraphUnpool(hidden_dim)

        self.s1_l1_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)
        self.s1_l2_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)
        self.s1_l3_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)

        self.s2_l1_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)
        self.s2_l2_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)
        self.s2_l3_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)

        self.s3_l1_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)
        self.s3_l2_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)
        self.s3_l3_gcn = TAGConvWithDropout(hidden_dim, hidden_dim)

        if num_cross_layers >= 1:
            self.pool_s12_1 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s21_1 = GraphUnpool(hidden_dim)
            self.pool_s23_1 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s32_1 = GraphUnpool(hidden_dim)
        if num_cross_layers >= 2:
            self.pool_s12_2 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s21_2 = GraphUnpool(hidden_dim)
            self.pool_s23_2 = GraphPool(hidden_dim, use_gcn=True)
            self.unpool_s32_2 = GraphUnpool(hidden_dim)

        self.cross_feature_1 = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.cross_feature_1.weight)

        self.cross_feature_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        torch.nn.init.xavier_normal_(self.cross_feature_2.weight)

    def forward(self, graph, feat):
        # start of scale-1
        graph_scale1 = graph
        feat_scale1 = self.start_gcn_scale1(graph_scale1, feat)

        feat_origin = feat_scale1
        feat_scale1_neg = feat_scale1[torch.randperm(feat_scale1.size(0))] # negative samples
        logit_s1, scores_s1, select_idx_s1, non_select_idx_s1, feat_down_s1 = \
            self.index_select_scale1(graph_scale1, feat_scale1, feat_scale1_neg)

        feat_scale2, graph_scale2 = self.start_pool_s12(graph_scale1, feat_scale1,
                                                        select_idx_s1, non_select_idx_s1,
                                                        scores_s1, pool_graph=True)
        
        # start of scale-2
        feat_scale2 = self.start_gcn_scale2(graph_scale2, feat_scale2)
        feat_scale2_neg = feat_scale2[torch.randperm(feat_scale2.size(0))] # negative samples
        logit_s2, scores_s2, select_idx_s2, non_select_idx_s2, feat_down_s2 = \
            self.index_select_scale2(graph_scale2, feat_scale2, feat_scale2_neg)


        feat_scale3, graph_scale3 = self.start_pool_s23(graph_scale2, feat_scale2,
                                                        select_idx_s2, non_select_idx_s2,
                                                        scores_s2, pool_graph=True)
        
        # layer-1
        res_s1_0, res_s2_0, res_s3_0 = feat_scale1, feat_scale2, feat_scale3
        
        feat_scale1 = F.relu(self.s1_l1_gcn(graph_scale1, feat_scale1))
        feat_scale2 = F.relu(self.s2_l1_gcn(graph_scale2, feat_scale2))
        feat_scale3 = F.relu(self.s3_l1_gcn(graph_scale3, feat_scale3))

        if self.num_cross_layers >= 1:
            feat_s12_fu = self.pool_s12_1(graph_scale1, feat_scale1,
                                          select_idx_s1, non_select_idx_s1,
                                          scores_s1)
            feat_s21_fu = self.unpool_s21_1(graph_scale1, feat_scale2, select_idx_s1)
            feat_s23_fu = self.pool_s23_1(graph_scale2, feat_scale2,
                                          select_idx_s2, non_select_idx_s2,
                                          scores_s2)
            feat_s32_fu = self.unpool_s32_1(graph_scale2, feat_scale3, select_idx_s2)

            feat_all = torch.cat((feat_scale1, feat_scale2, feat_scale3), 0)
            feat_scale_all = self.cross_feature_1(feat_all)
            feat_scale1, feat_scale2, feat_scale3 = torch.split(feat_scale_all, [feat_scale1.size()[0], feat_scale2.size()[0], feat_scale3.size()[0]], 0)
        
        # layer-2
        feat_scale1 = F.relu(self.s1_l2_gcn(graph_scale1, feat_scale1))
        feat_scale2 = F.relu(self.s2_l2_gcn(graph_scale2, feat_scale2))
        feat_scale3 = F.relu(self.s3_l2_gcn(graph_scale3, feat_scale3))

        # final layers
        feat_s3_out = self.end_unpool_s32(graph_scale2, feat_scale3, select_idx_s2) + feat_down_s2
        feat_s2_out = self.end_unpool_s21(graph_scale1, feat_scale2 + feat_s3_out, select_idx_s1)
        feat_agg = feat_scale1 + self.fuse_weight * feat_s2_out + self.fuse_weight * feat_down_s1
        feat_agg = torch.cat((feat_agg, feat_origin), dim=1)
        feat_agg = self.end_gcn(graph_scale1, feat_agg)

        return feat_agg, logit_s1, logit_s2

class GraphClassifier(torch.nn.Module):
    """
    Description
    -----------
    Graph Classifier for graph classification.
    GXN + MLP
    """
    def __init__(self, args):
        super(GraphClassifier, self).__init__()
        self.graph_type = args.graph_type
        self.magnn = MAGNN(in_feats = {'I': 300, 'S':300, 'A':512},
                                        h_feats = 300, inter_attn_feats = 128, num_heads = 1, 
                                        num_classes = 2, num_layers = 2,  metapath_list=['IUS', 'AUS', 'AWI', 'AXA', 'IYI', 'SZS'],
                                        edge_type_list=['S-I', 'I-S'], dropout_rate = 0.5, encoder_type = 'Average', activation = F.elu)
        self.gxn = GraphCrossModule(args.pool_ratios, in_dim=args.in_dim, out_dim=args.embed_dim,
                                    hidden_dim=args.hidden_dim//2, cross_weight=args.cross_weight,
                                    fuse_weight=args.fuse_weight, dist=1)
        self.readout_nodes = 10
        self.sortpool = SortPooling(self.readout_nodes)
        self.lin1 = torch.nn.Linear(self.readout_nodes*args.embed_dim, args.out_dim)

    def forward(self, graph:DGLGraph, node_feat:Tensor, edge_feat:Optional[Tensor]=None):
        if self.graph_type == "heterogeneous":
            myfeature_dict={}
            if 'S' in graph.ntypes:
                myfeature_dict['S'] = graph.nodes['S'].data['embedding']
            if 'I' in graph.ntypes:
                myfeature_dict['I'] = graph.nodes['I'].data['embedding']
            if 'A' in graph.ntypes:
                myfeature_dict['A'] = graph.nodes['A'].data['embedding']
            new_homo_graph = self.magnn(graph, myfeature_dict)
            embed_feat, logits1, logits2 = self.gxn(new_homo_graph, new_homo_graph.ndata["feat"])
            batch_sortpool_feats = self.sortpool(new_homo_graph, embed_feat)
        else:
            embed_feat, logits1, logits2 = self.gxn(graph, node_feat)
            batch_sortpool_feats = self.sortpool(graph, embed_feat)
        # print(embed_feat.size(), "embed_feat.size()")
        # print(batch_sortpool_feats.size(), "batch_sortpool_feats.size()")
        logits = self.lin1(batch_sortpool_feats)
        # print(logits.size(), "logits,size()")
        return F.log_softmax(logits, dim=1), batch_sortpool_feats, logits1, logits2
