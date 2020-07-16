##################################################################
# From my_ggnn_16: no knowledge
##################################################################

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import pickle
from lib.my_util import MLP

def wrap(nparr):
    return Variable(torch.from_numpy(nparr).float().cuda(), requires_grad=False)

def arange(num):
    return torch.arange(num).type(torch.LongTensor).cuda()

def normalize(tensor, dim, eps=1e-4):
    return tensor / torch.sqrt(torch.max((tensor**2).sum(dim=dim, keepdim=True), wrap(np.asarray([eps]))))

class GGNN(nn.Module):
    def __init__(self, emb_path, graph_path, time_step_num=3, hidden_dim=512, output_dim=512, 
                 use_embedding=True, use_knowledge=True, refine_obj_cls=False, top_k_to_keep=5, normalize_messages=True):
        super(GGNN, self).__init__()
        self.time_step_num = time_step_num
                
        self.fc_mp_send_img_ent = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        self.fc_mp_send_img_pred = MLP([hidden_dim, hidden_dim // 2, hidden_dim // 4], act_fn='ReLU', last_act=True)
        
        self.fc_mp_receive_img_ent = MLP([2 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        self.fc_mp_receive_img_pred = MLP([2 * hidden_dim // 4, 3 * hidden_dim // 4, hidden_dim], act_fn='ReLU', last_act=True)
        
        self.fc_eq3_w_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_ent = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_img_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_img_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_img_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_img_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_img_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_img_pred = nn.Linear(hidden_dim, hidden_dim)

        self.fc_output_proj_img_pred = MLP([hidden_dim, hidden_dim, 51], act_fn='ReLU', last_act=False)
        
        self.refine_obj_cls = refine_obj_cls
        if self.refine_obj_cls:
            self.fc_output_proj_img_ent = MLP([hidden_dim, hidden_dim, 151], act_fn='ReLU', last_act=False)
        
        self.debug_info = {}
        self.top_k_to_keep = top_k_to_keep
        self.normalize_messages = normalize_messages
        
    def forward(self, rel_inds, ent_cls_logits, obj_fmaps, vr):
        num_img_ent = ent_cls_logits.size(0)
        num_img_pred = rel_inds.size(0)
        num_ont_ent = 151
        num_ont_pred = 51

        nodes_img_ent = obj_fmaps
        nodes_img_pred = vr
        
        edges_img_pred2subj = wrap(np.zeros((num_img_pred, num_img_ent)))
        edges_img_pred2subj[arange(num_img_pred), rel_inds[:, 0]] = 1
        edges_img_pred2obj = wrap(np.zeros((num_img_pred, num_img_ent)))
        edges_img_pred2obj[arange(num_img_pred), rel_inds[:, 1]] = 1
        edges_img_subj2pred = edges_img_pred2subj.t()
        edges_img_obj2pred = edges_img_pred2obj.t()

        edges_img_pred2subj = edges_img_pred2subj / torch.max(edges_img_pred2subj.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        edges_img_pred2obj = edges_img_pred2obj / torch.max(edges_img_pred2obj.sum(dim=0, keepdim=True), wrap(np.asarray([1.0])))
        
        for t in range(self.time_step_num):
            message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent)
            message_send_img_pred = self.fc_mp_send_img_pred(nodes_img_pred)
            
            message_incoming_img_ent = torch.stack([
                torch.mm(edges_img_pred2subj.t(), message_send_img_pred),
                torch.mm(edges_img_pred2obj.t(), message_send_img_pred),
            ], 1)
            
            message_incoming_img_pred = torch.stack([
                torch.mm(edges_img_subj2pred.t(), message_send_img_ent),
                torch.mm(edges_img_obj2pred.t(), message_send_img_ent),
            ], 1)
                        
            if self.normalize_messages:
                message_incoming_img_ent = normalize(message_incoming_img_ent, 2)
                message_incoming_img_pred = normalize(message_incoming_img_pred, 2)
            
            message_received_img_ent = self.fc_mp_receive_img_ent(message_incoming_img_ent.view(num_img_ent, -1))            
            message_received_img_pred = self.fc_mp_receive_img_pred(message_incoming_img_pred.view(num_img_pred, -1))

            z_img_ent = torch.sigmoid(self.fc_eq3_w_img_ent(message_received_img_ent) + self.fc_eq3_u_img_ent(nodes_img_ent))
            r_img_ent = torch.sigmoid(self.fc_eq4_w_img_ent(message_received_img_ent) + self.fc_eq4_u_img_ent(nodes_img_ent))
            h_img_ent = torch.tanh(self.fc_eq5_w_img_ent(message_received_img_ent) + self.fc_eq5_u_img_ent(r_img_ent * nodes_img_ent))
            nodes_img_ent_new = (1 - z_img_ent) * nodes_img_ent + z_img_ent * h_img_ent

            z_img_pred = torch.sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred) + self.fc_eq3_u_img_pred(nodes_img_pred))
            r_img_pred = torch.sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
            h_img_pred = torch.tanh(self.fc_eq5_w_img_pred(message_received_img_pred) + self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred))
            nodes_img_pred_new = (1 - z_img_pred) * nodes_img_pred + z_img_pred * h_img_pred

            nodes_img_ent = nodes_img_ent_new
            nodes_img_pred = nodes_img_pred_new
            
            
        pred_cls_logits = self.fc_output_proj_img_pred(nodes_img_pred)

        if self.refine_obj_cls:
            ent_cls_logits = self.fc_output_proj_img_ent(nodes_img_ent)
                
        return pred_cls_logits, ent_cls_logits

