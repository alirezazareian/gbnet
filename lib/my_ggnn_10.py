##################################################################
# From my_ggnn_09: Dynamically connecting entities to ontology too
# Also a minor change: img2ont edges are now normalized over ont rather than img
##################################################################

import os, sys
import torch
from torch import tensor as torch_tensor, float32 as torch_float32, \
    LongTensor as torch_LongTensor, arange as torch_arange, mm as torch_mm, \
    zeros as torch_zeros, \
    sigmoid as torch_sigmoid, tanh as torch_tanh, cat as torch_cat, \
    sum as torch_sum, abs as torch_abs
from torch.nn import Module, Linear
from torch.nn.functional import softmax as F_softmax
import numpy as np
import pickle
from os import environ as os_environ
from lib.my_util import MLP

def wrap(nparr):
    return torch_tensor(nparr, dtype=torch_float32, device=int(os_environ['CUDA_VISIBLE_DEVICES']), requires_grad=False)

def arange(num):
    return torch_arange(num, dtype=torch_LongTensor, device=int(os_environ['CUDA_VISIBLE_DEVICES']))

class GGNN(Module):
    def __init__(self, emb_path, graph_path, time_step_num=3, hidden_dim=512, output_dim=512,
                 use_embedding=True, use_knowledge=True, refine_obj_cls=False, num_ents=151, num_preds=51, config=None):
        super(GGNN, self).__init__()
        self.time_step_num = time_step_num

        if use_embedding:
            with open(emb_path, 'rb') as fin:
                self.emb_ent, self.emb_pred = pickle.load(fin)
            self.emb_ent = wrap(self.emb_ent)
            self.emb_pred = wrap(self.emb_pred)
        else:
            # self.emb_ent = np.eye(num_ents, dtype=np.float32)
            self.emb_ent = torch_eye(num_ents, dtype=torch_float32)
            # self.emb_pred = np.eye(num_preds, dtype=np.float32)
            self.emb_pred = torch_eye(num_preds, dtype=torch_float32)

        num_ont_ent = self.emb_ent.size(0)
        num_ont_pred = self.emb_pred.size(0)

        if use_knowledge:
            with open(graph_path, 'rb') as fin:
                edge_dict = pickle.load(fin)
            self.adjmtx_ent2ent = edge_dict['edges_ent2ent']
            self.adjmtx_ent2pred = edge_dict['edges_ent2pred']
            self.adjmtx_pred2ent = edge_dict['edges_pred2ent']
            self.adjmtx_pred2pred = edge_dict['edges_pred2pred']
        else:
            self.adjmtx_ent2ent = np.zeros((1, num_ents, num_ents), dtype=np.float32)
            self.adjmtx_ent2pred = np.zeros((1, num_ents, num_preds), dtype=np.float32)
            self.adjmtx_pred2ent = np.zeros((1, num_preds, num_ents), dtype=np.float32)
            self.adjmtx_pred2pred = np.zeros((1, num_preds, num_preds), dtype=np.float32)

        self.edges_ont_ent2ent = wrap(self.adjmtx_ent2ent)
        self.edges_ont_ent2pred = wrap(self.adjmtx_ent2pred)
        self.edges_ont_pred2ent = wrap(self.adjmtx_pred2ent)
        self.edges_ont_pred2pred = wrap(self.adjmtx_pred2pred)

        self.num_edge_types_ent2ent = self.adjmtx_ent2ent.shape[0]
        self.num_edge_types_ent2pred = self.adjmtx_ent2pred.shape[0]
        self.num_edge_types_pred2ent = self.adjmtx_pred2ent.shape[0]
        self.num_edge_types_pred2pred = self.adjmtx_pred2pred.shape[0]

        self.fc_init_ont_ent = Linear(self.emb_ent.size(1), hidden_dim)
        self.fc_init_ont_pred = Linear(self.emb_pred.size(1), hidden_dim)

        self.fc_mp_send_ont_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_ont_pred = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_pred = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)

        self.fc_mp_receive_ont_ent = MLP([(self.num_edge_types_ent2ent + self.num_edge_types_pred2ent + 1) * hidden_dim // 4,
                                          (self.num_edge_types_ent2ent + self.num_edge_types_pred2ent + 1) * hidden_dim // 4,
                                          hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_ont_pred = MLP([(self.num_edge_types_ent2pred + self.num_edge_types_pred2pred + 1) * hidden_dim // 4,
                                           (self.num_edge_types_ent2pred + self.num_edge_types_pred2pred + 1) * hidden_dim // 4,
                                           hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_ent = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_pred = MLP([3 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)

        self.fc_eq3_w_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_ent = Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_pred = Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_ent = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_ent = Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_pred = Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_pred = Linear(hidden_dim, hidden_dim)

        self.fc_output_proj_img_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        self.fc_output_proj_ont_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

        self.refine_obj_cls = refine_obj_cls
        if self.refine_obj_cls:
            self.fc_output_proj_img_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
            self.fc_output_proj_ont_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

        self.debug_info = {}

        self.with_clean_classifier = config.MODEL.ROI_RELATION_HEAD.WITH_CLEAN_CLASSIFIER
        self.with_transfer = config.MODEL.ROI_RELATION_HEAD.WITH_TRANSFER_CLASSIFIER

        if self.with_clean_classifier:
            self.fc_output_proj_img_pred_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
            self.fc_output_proj_ont_pred_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

            if self.refine_obj_cls:
                self.fc_output_proj_img_ent_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
                self.fc_output_proj_ont_ent_clean = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)

            if self.with_transfer:
                self.devices = config.MODEL.DEVICE
                print("!!!!!!!!!With Confusion Matrix Channel!!!!!")
                pred_adj_np = np.load(config.MODEL.CONF_MAT_FREQ_TRAIN)
                # pred_adj_np = 1.0 - pred_adj_np
                pred_adj_np[0, :] = 0.0
                pred_adj_np[:, 0] = 0.0
                pred_adj_np[0, 0] = 1.0
                # adj_i_j means the baseline outputs category j, but the ground truth is i.
                pred_adj_np = pred_adj_np / (pred_adj_np.sum(-1)[:, None] + 1e-8)
                self.pred_adj_nor = torch_tensor(pred_adj_np, dtype=torch_float32, decie=self.devices)


    def forward(self, rel_inds, obj_probs, obj_fmaps, vr):
        num_img_ent = obj_probs.size(0)
        num_img_pred = rel_inds.size(0)

        debug_info = self.debug_info
        debug_info['rel_inds'] = rel_inds
        debug_info['obj_probs'] = obj_probs

        refine_obj_cls = self.refine_obj_cls

        nodes_ont_ent = self.fc_init_ont_ent(self.emb_ent)
        nodes_ont_pred = self.fc_init_ont_pred(self.emb_pred)
        nodes_img_ent = obj_fmaps
        nodes_img_pred = vr

        # edges_img_pred2subj = wrap(np.zeros((num_img_pred, num_img_ent)))
        edges_img_pred2subj = torch_zeros((num_img_pred, num_img_ent), dtype=torch_float32, devices=int(os_environ['CUDA_VISIBLE_DEVICES']), requires_grad=False)
        edges_img_pred2subj[arange(num_img_pred), rel_inds[:, 0]] = 1
        # edges_img_pred2obj = wrap(np.zeros((num_img_pred, num_img_ent)))
        edges_img_pred2obj = torch_zeros((num_img_pred, num_img_ent), dtype=torch_float32, devices=int(os_environ['CUDA_VISIBLE_DEVICES']), requires_grad=False)
        edges_img_pred2obj[arange(num_img_pred), rel_inds[:, 1]] = 1
        edges_img_subj2pred = edges_img_pred2subj.t()
        edges_img_obj2pred = edges_img_pred2obj.t()

        # edges_img2ont_ent = wrap(obj_probs.data.cpu().numpy())
        edges_img2ont_ent = torch_tensor(obj_probs.data, dtype=torch_float32, devices=int(os_environ['CUDA_VISIBLE_DEVICES']), requires_grad=False)
        # edges_img2ont_ent = obj_probs.detach()
        edges_ont2img_ent = edges_img2ont_ent.t()

        # edges_img2ont_pred = wrap(np.zeros((num_img_pred, num_ont_pred)))
        edges_img2ont_pred = torch_zeros((num_img_pred, num_ont_pred))
        edges_ont2img_pred = edges_img2ont_pred.t()

        ent_cls_logits = None

        edges_ont_ent2ent = self.edges_ont_ent2ent
        edges_ont_pred2ent = self.edges_ont_pred2ent
        edges_ont_ent2pred = self.edges_ont_ent2pred
        edges_ont_pred2pred = self.edges_ont_pred2pred

        num_edge_types_ent2ent = self.num_edge_types_ent2ent
        num_edge_types_pred2ent = self.num_edge_types_pred2ent
        num_edge_types_ent2pred = self.num_edge_types_ent2pred
        num_edge_types_pred2pred = self.num_edge_types_pred2pred

        with_clean_classifier = self.with_clean_classifier
        with_transfer = self.with_transfer

        if with_clean_classifier and with_transfer:
            pred_adj_nor = self.pred_adj_nor

        for t in range(self.time_step_num):
            message_send_ont_ent = self.fc_mp_send_ont_ent(nodes_ont_ent)
            message_send_ont_pred = self.fc_mp_send_ont_pred(nodes_ont_pred)
            message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent)
            message_send_img_pred = self.fc_mp_send_img_pred(nodes_img_pred)

            message_received_ont_ent = self.fc_mp_receive_ont_ent(torch_cat(
                [torch_mm(edges_ont_ent2ent[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2ent)] +
                [torch_mm(edges_ont_pred2ent[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2ent)] +
                # [torch_mm(edges_img2ont_ent.t(), message_send_img_ent),]
                [torch_mm(edges_ont2img_ent, message_send_img_ent),]
            , 1))

            message_received_ont_pred = self.fc_mp_receive_ont_pred(torch_cat(
                [torch_mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) for i in range(num_edge_types_ent2pred)] +
                [torch_mm(edges_ont_pred2pred[i].t(), message_send_ont_pred) for i in range(num_edge_types_pred2pred)] +
                # [torch_mm(edges_img2ont_pred.t(), message_send_img_pred),]
                [torch_mm(edges_ont2img_pred, message_send_img_pred),]
            , 1))

            message_received_img_ent = self.fc_mp_receive_img_ent(torch_cat([
                # torch_mm(edges_img_pred2subj.t(), message_send_img_pred),
                torch_mm(edges_img_subj2pred, message_send_img_pred),
                # torch_mm(edges_img_pred2obj.t(), message_send_img_pred),
                torch_mm(edges_img_obj2pred, message_send_img_pred),
                # torch_mm(edges_ont2img_ent.t(), message_send_ont_ent),
                torch_mm(edges_img2ont_ent, message_send_ont_ent),
            ], 1))

            message_received_img_pred = self.fc_mp_receive_img_pred(torch_cat([
                # torch_mm(edges_img_subj2pred.t(), message_send_img_ent),
                torch_mm(edges_img_pred2subj, message_send_img_ent),
                # torch_mm(edges_img_obj2pred.t(), message_send_img_ent),
                torch_mm(edges_img_pred2obj, message_send_img_ent),
                # torch_mm(edges_ont2img_pred.t(), message_send_ont_pred),
                torch_mm(edges_img2ont_pred, message_send_ont_pred),
            ], 1))

            z_ont_ent = torch_sigmoid(self.fc_eq3_w_ont_ent(message_received_ont_ent) + self.fc_eq3_u_ont_ent(nodes_ont_ent))
            r_ont_ent = torch_sigmoid(self.fc_eq4_w_ont_ent(message_received_ont_ent) + self.fc_eq4_u_ont_ent(nodes_ont_ent))
            h_ont_ent = torch_tanh(self.fc_eq5_w_ont_ent(message_received_ont_ent) + self.fc_eq5_u_ont_ent(r_ont_ent * nodes_ont_ent))
            nodes_ont_ent_new = (1 - z_ont_ent) * nodes_ont_ent + z_ont_ent * h_ont_ent

            z_ont_pred = torch_sigmoid(self.fc_eq3_w_ont_pred(message_received_ont_pred) + self.fc_eq3_u_ont_pred(nodes_ont_pred))
            r_ont_pred = torch_sigmoid(self.fc_eq4_w_ont_pred(message_received_ont_pred) + self.fc_eq4_u_ont_pred(nodes_ont_pred))
            h_ont_pred = torch_tanh(self.fc_eq5_w_ont_pred(message_received_ont_pred) + self.fc_eq5_u_ont_pred(r_ont_pred * nodes_ont_pred))
            nodes_ont_pred_new = (1 - z_ont_pred) * nodes_ont_pred + z_ont_pred * h_ont_pred

            z_img_ent = torch_sigmoid(self.fc_eq3_w_img_ent(message_received_img_ent) + self.fc_eq3_u_img_ent(nodes_img_ent))
            r_img_ent = torch_sigmoid(self.fc_eq4_w_img_ent(message_received_img_ent) + self.fc_eq4_u_img_ent(nodes_img_ent))
            h_img_ent = torch_tanh(self.fc_eq5_w_img_ent(message_received_img_ent) + self.fc_eq5_u_img_ent(r_img_ent * nodes_img_ent))
            nodes_img_ent_new = (1 - z_img_ent) * nodes_img_ent + z_img_ent * h_img_ent

            z_img_pred = torch_sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred) + self.fc_eq3_u_img_pred(nodes_img_pred))
            r_img_pred = torch_sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
            h_img_pred = torch_tanh(self.fc_eq5_w_img_pred(message_received_img_pred) + self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred))
            nodes_img_pred_new = (1 - z_img_pred) * nodes_img_pred + z_img_pred * h_img_pred

            relative_state_change_ont_ent = torch_sum(torch_abs(nodes_ont_ent_new - nodes_ont_ent)) / torch_sum(torch_abs(nodes_ont_ent))
            relative_state_change_ont_pred = torch_sum(torch_abs(nodes_ont_pred_new - nodes_ont_pred)) / torch_sum(torch_abs(nodes_ont_pred))
            relative_state_change_img_ent = torch_sum(torch_abs(nodes_img_ent_new - nodes_img_ent)) / torch_sum(torch_abs(nodes_img_ent))
            relative_state_change_img_pred = torch_sum(torch_abs(nodes_img_pred_new - nodes_img_pred)) / torch_sum(torch_abs(nodes_img_pred))

            debug_info[f'relative_state_change_{t}'] = [relative_state_change_ont_ent, relative_state_change_ont_pred, relative_state_change_img_ent, relative_state_change_img_pred]

            nodes_ont_ent = nodes_ont_ent_new
            nodes_ont_pred = nodes_ont_pred_new
            nodes_img_ent = nodes_img_ent_new
            nodes_img_pred = nodes_img_pred_new

            pred_cls_logits = torch_mm(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred).t())
            edges_img2ont_pred = F_softmax(pred_cls_logits, dim=1)
            edges_ont2img_pred = edges_img2ont_pred.t()

            if refine_obj_cls:
                ent_cls_logits = torch_mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
                edges_img2ont_ent = F_softmax(ent_cls_logits, dim=1)
                edges_ont2img_ent = edges_img2ont_ent.t()

            if with_clean_classifier:
                pred_cls_logits_clean = torch_mm(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred_clean(nodes_ont_pred).t())
                if with_transfer:
                    pred_cls_logits_clean = (pred_adj_nor @ pred_cls_logits_clean.T).T

                pred_cls_logits = pred_cls_logits_clean


        return pred_cls_logits, ent_cls_logits
