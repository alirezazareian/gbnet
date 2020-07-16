##################################################################
# From my_ggnn_09: Dynamically connecting entities to ontology too
# Also a minor change: img2ont edges are now normalized over ont rather than img
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

class GGNN(nn.Module):
    def __init__(self, emb_path, graph_path, time_step_num=3, hidden_dim=512, output_dim=512, 
                 use_embedding=True, use_knowledge=True, refine_obj_cls=False):
        super(GGNN, self).__init__()
        self.time_step_num = time_step_num
                
        if use_embedding:
            with open(emb_path, 'rb') as fin:
                self.emb_ent, self.emb_pred = pickle.load(fin)
        else:
            self.emb_ent = np.eye(151, dtype=np.float32)
            self.emb_pred = np.eye(51, dtype=np.float32)

        if use_knowledge:
            with open(graph_path, 'rb') as fin:
                edge_dict = pickle.load(fin)
            self.adjmtx_ent2ent = edge_dict['edges_ent2ent']
            self.adjmtx_ent2pred = edge_dict['edges_ent2pred']
            self.adjmtx_pred2ent = edge_dict['edges_pred2ent']
            self.adjmtx_pred2pred = edge_dict['edges_pred2pred']
        else:
            self.adjmtx_ent2ent = np.zeros((1, 151, 151), dtype=np.float32)
            self.adjmtx_ent2pred = np.zeros((1, 151, 51), dtype=np.float32)
            self.adjmtx_pred2ent = np.zeros((1, 51, 151), dtype=np.float32)
            self.adjmtx_pred2pred = np.zeros((1, 51, 51), dtype=np.float32)
        
        self.num_edge_types_ent2ent = self.adjmtx_ent2ent.shape[0]
        self.num_edge_types_ent2pred = self.adjmtx_ent2pred.shape[0]
        self.num_edge_types_pred2ent = self.adjmtx_pred2ent.shape[0]
        self.num_edge_types_pred2pred = self.adjmtx_pred2pred.shape[0]
        
        self.fc_init_ont_ent = nn.Linear(self.emb_ent.shape[1], hidden_dim)
        self.fc_init_ont_pred = nn.Linear(self.emb_pred.shape[1], hidden_dim)
        
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
        
        self.fc_eq3_w_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_ent = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_ent = nn.Linear(hidden_dim, hidden_dim)

        self.fc_eq3_w_ont_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq3_u_ont_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_w_ont_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq4_u_ont_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_w_ont_pred = nn.Linear(hidden_dim, hidden_dim)
        self.fc_eq5_u_ont_pred = nn.Linear(hidden_dim, hidden_dim)

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

        self.fc_output_proj_img_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        self.fc_output_proj_ont_pred = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
        
        self.refine_obj_cls = refine_obj_cls
        if self.refine_obj_cls:
            self.fc_output_proj_img_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)
            self.fc_output_proj_ont_ent = MLP([hidden_dim, hidden_dim, hidden_dim], act_fn='ReLU', last_act=False)            
        
        self.debug_info = {}
        
        
    def forward(self, rel_inds, obj_probs, obj_fmaps, vr):
        num_img_ent = obj_probs.size(0)
        num_img_pred = rel_inds.size(0)
        num_ont_ent = self.emb_ent.shape[0]
        num_ont_pred = self.emb_pred.shape[0]

        self.debug_info['rel_inds'] = rel_inds
        self.debug_info['obj_probs'] = obj_probs
        
        nodes_ont_ent = self.fc_init_ont_ent(wrap(self.emb_ent))
        nodes_ont_pred = self.fc_init_ont_pred(wrap(self.emb_pred))        
        nodes_img_ent = obj_fmaps
        nodes_img_pred = vr
        
        edges_ont_ent2ent = wrap(self.adjmtx_ent2ent)
        edges_ont_ent2pred = wrap(self.adjmtx_ent2pred)
        edges_ont_pred2ent = wrap(self.adjmtx_pred2ent)
        edges_ont_pred2pred = wrap(self.adjmtx_pred2pred)

        edges_img_pred2subj = wrap(np.zeros((num_img_pred, num_img_ent)))
        edges_img_pred2subj[arange(num_img_pred), rel_inds[:, 0]] = 1
        edges_img_pred2obj = wrap(np.zeros((num_img_pred, num_img_ent)))
        edges_img_pred2obj[arange(num_img_pred), rel_inds[:, 1]] = 1
        edges_img_subj2pred = edges_img_pred2subj.t()
        edges_img_obj2pred = edges_img_pred2obj.t()
        
        edges_img2ont_ent = wrap(obj_probs.data.cpu().numpy())
        edges_ont2img_ent = edges_img2ont_ent.t()

        edges_img2ont_pred =  wrap(np.zeros((num_img_pred, num_ont_pred)))
        edges_ont2img_pred = edges_img2ont_pred.t()
        
        ent_cls_logits = None
        
        for t in range(self.time_step_num):
            message_send_ont_ent = self.fc_mp_send_ont_ent(nodes_ont_ent)
            message_send_ont_pred = self.fc_mp_send_ont_pred(nodes_ont_pred)
            message_send_img_ent = self.fc_mp_send_img_ent(nodes_img_ent)
            message_send_img_pred = self.fc_mp_send_img_pred(nodes_img_pred)
            
            message_received_ont_ent = self.fc_mp_receive_ont_ent(torch.cat(
                [torch.mm(edges_ont_ent2ent[i].t(), message_send_ont_ent) for i in range(self.num_edge_types_ent2ent)] +
                [torch.mm(edges_ont_pred2ent[i].t(), message_send_ont_pred) for i in range(self.num_edge_types_pred2ent)] +
                [torch.mm(edges_img2ont_ent.t(), message_send_img_ent),]
            , 1))
            
            message_received_ont_pred = self.fc_mp_receive_ont_pred(torch.cat(
                [torch.mm(edges_ont_ent2pred[i].t(), message_send_ont_ent) for i in range(self.num_edge_types_ent2pred)] +
                [torch.mm(edges_ont_pred2pred[i].t(), message_send_ont_pred) for i in range(self.num_edge_types_pred2pred)] +
                [torch.mm(edges_img2ont_pred.t(), message_send_img_pred),]
            , 1))
            
            message_received_img_ent = self.fc_mp_receive_img_ent(torch.cat([
                torch.mm(edges_img_pred2subj.t(), message_send_img_pred),
                torch.mm(edges_img_pred2obj.t(), message_send_img_pred),
                torch.mm(edges_ont2img_ent.t(), message_send_ont_ent),
            ], 1))
            
            message_received_img_pred = self.fc_mp_receive_img_pred(torch.cat([
                torch.mm(edges_img_subj2pred.t(), message_send_img_ent),
                torch.mm(edges_img_obj2pred.t(), message_send_img_ent),
                torch.mm(edges_ont2img_pred.t(), message_send_ont_pred),
            ], 1))
            
            z_ont_ent = torch.sigmoid(self.fc_eq3_w_ont_ent(message_received_ont_ent) + self.fc_eq3_u_ont_ent(nodes_ont_ent))
            r_ont_ent = torch.sigmoid(self.fc_eq4_w_ont_ent(message_received_ont_ent) + self.fc_eq4_u_ont_ent(nodes_ont_ent))
            h_ont_ent = torch.tanh(self.fc_eq5_w_ont_ent(message_received_ont_ent) + self.fc_eq5_u_ont_ent(r_ont_ent * nodes_ont_ent))
            nodes_ont_ent_new = (1 - z_ont_ent) * nodes_ont_ent + z_ont_ent * h_ont_ent

            z_ont_pred = torch.sigmoid(self.fc_eq3_w_ont_pred(message_received_ont_pred) + self.fc_eq3_u_ont_pred(nodes_ont_pred))
            r_ont_pred = torch.sigmoid(self.fc_eq4_w_ont_pred(message_received_ont_pred) + self.fc_eq4_u_ont_pred(nodes_ont_pred))
            h_ont_pred = torch.tanh(self.fc_eq5_w_ont_pred(message_received_ont_pred) + self.fc_eq5_u_ont_pred(r_ont_pred * nodes_ont_pred))
            nodes_ont_pred_new = (1 - z_ont_pred) * nodes_ont_pred + z_ont_pred * h_ont_pred

            z_img_ent = torch.sigmoid(self.fc_eq3_w_img_ent(message_received_img_ent) + self.fc_eq3_u_img_ent(nodes_img_ent))
            r_img_ent = torch.sigmoid(self.fc_eq4_w_img_ent(message_received_img_ent) + self.fc_eq4_u_img_ent(nodes_img_ent))
            h_img_ent = torch.tanh(self.fc_eq5_w_img_ent(message_received_img_ent) + self.fc_eq5_u_img_ent(r_img_ent * nodes_img_ent))
            nodes_img_ent_new = (1 - z_img_ent) * nodes_img_ent + z_img_ent * h_img_ent

            z_img_pred = torch.sigmoid(self.fc_eq3_w_img_pred(message_received_img_pred) + self.fc_eq3_u_img_pred(nodes_img_pred))
            r_img_pred = torch.sigmoid(self.fc_eq4_w_img_pred(message_received_img_pred) + self.fc_eq4_u_img_pred(nodes_img_pred))
            h_img_pred = torch.tanh(self.fc_eq5_w_img_pred(message_received_img_pred) + self.fc_eq5_u_img_pred(r_img_pred * nodes_img_pred))
            nodes_img_pred_new = (1 - z_img_pred) * nodes_img_pred + z_img_pred * h_img_pred

            relative_state_change_ont_ent = torch.sum(torch.abs(nodes_ont_ent_new - nodes_ont_ent)) / torch.sum(torch.abs(nodes_ont_ent))
            relative_state_change_ont_pred = torch.sum(torch.abs(nodes_ont_pred_new - nodes_ont_pred)) / torch.sum(torch.abs(nodes_ont_pred))
            relative_state_change_img_ent = torch.sum(torch.abs(nodes_img_ent_new - nodes_img_ent)) / torch.sum(torch.abs(nodes_img_ent))
            relative_state_change_img_pred = torch.sum(torch.abs(nodes_img_pred_new - nodes_img_pred)) / torch.sum(torch.abs(nodes_img_pred))
        
            self.debug_info[f'relative_state_change_{t}'] = [relative_state_change_ont_ent, relative_state_change_ont_pred, relative_state_change_img_ent, relative_state_change_img_pred]
        
            nodes_ont_ent = nodes_ont_ent_new
            nodes_ont_pred = nodes_ont_pred_new
            nodes_img_ent = nodes_img_ent_new
            nodes_img_pred = nodes_img_pred_new
            
            pred_cls_logits = torch.mm(self.fc_output_proj_img_pred(nodes_img_pred), self.fc_output_proj_ont_pred(nodes_ont_pred).t())
            edges_img2ont_pred = F.softmax(pred_cls_logits, dim=1)
            edges_ont2img_pred = edges_img2ont_pred.t()
            
            if self.refine_obj_cls:
                ent_cls_logits = torch.mm(self.fc_output_proj_img_ent(nodes_img_ent), self.fc_output_proj_ont_ent(nodes_ont_ent).t())
                edges_img2ont_ent = F.softmax(ent_cls_logits, dim=1)
                edges_ont2img_ent = edges_img2ont_ent.t()
                
        return pred_cls_logits, ent_cls_logits

