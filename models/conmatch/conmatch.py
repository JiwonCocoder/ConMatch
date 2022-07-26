
import pickle
import sys
import pdb
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from collections import Counter
import os
import contextlib
from train_utils import AverageMeter

from .conmatch2_utils import consistency_loss as consistency_loss_fix
from .conmatch2_utils import consistency_loss_con

from .conmatch2_utils import confidence_loss, Get_Scalar

from train_utils import ce_loss, wd_loss, EMA, Bn_Controller
from scipy.special import softmax

from sklearn.metrics import *
from sklearn.metrics import classification_report
# from skimage.util import random_noise
from copy import deepcopy
import sys
import pdb
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


def custom_replace(tensor):
    # we create a copy of the original tensor, 
    # because of the way we are replacing them.
    res = tensor.clone()
    res[tensor==0] = 1
    res[tensor==1] = 0
    
    return res
    
def softmax_2(self, x, d = 1):
    r'''SFNet: Learning Object-aware Semantic Flow (Lee et al.)'''
    M, _ = x.max(dim=d, keepdim=True)
    x = x - M # subtract maximum value for stability
    exp_x = torch.exp(x)
    exp_x_sum = exp_x.sum(dim=d, keepdim=True)
    return exp_x / exp_x_sum

def split_con_input(con_input, num_lb, num_ulb, is_con_sup, is_con_unsup):
    #sup + unsup_w
    if is_con_sup == True and is_con_unsup == True:
        return con_input[:(num_lb + num_ulb)]
    #sup
    elif is_con_sup == True and is_con_unsup == False:
        return con_input[:num_lb]            
    #unsup_w
    elif is_con_sup == False and is_con_unsup == True:
        return con_input[num_lb:(num_lb + num_ulb)]
                    

def softmax_function(x):
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)
'''
(type_1) total: classification_reprot
'''
def estimate_metric_total(pred_cls, target, pred_val=None, thres=None):
    if pred_val is not None:
        con_pred = pred_val.ge(thres).long()
        con_true = (torch.eq(target.cuda(), pred_cls)).long()
        print(torch.count_nonzero(con_pred))
        print(classification_report(con_true.cpu().tolist(),con_pred.cpu().tolist(), target_names=['0', '1']))                
    else:
        num_class = 10
        num_clss_list = [str(i) for i in range(num_class)]
        print(classification_report(target.cpu().tolist(),pred_cls.cpu().tolist(), target_names=num_clss_list))                

'''
(type_2) individual: precision, recall, AUC (true, pred)
'''

def estimate_metric_binary(pred_cls, true_cls, pred_prob=None, thres=None):
    pred = pred_prob.ge(thres).long()
    true = (torch.eq(true_cls.cuda(), pred_cls)).long()
    precision = precision_score(true.cpu().tolist(), pred.cpu().tolist())               
    recall = recall_score(true.cpu().tolist(), pred.cpu().tolist())               
    # AUC = roc_auc_score(true.cpu().tolist(), pred_prob.cpu().tolist(), multi_class = 'ovo')               
    top1 = accuracy_score(true.cpu().tolist(), pred.cpu().tolist())
    return [precision, recall, top1]             
    # print(precision, recall)

def estimate_metric_binary_2(pred_cls, true_cls, pred_prob=None, thres=None, est_AUC = True):
    pred = pred_prob.ge(thres).long()
    true = (torch.eq(true_cls.cuda(), pred_cls.cuda())).long()
    precision = precision_score(true.cpu().tolist(), pred.cpu().tolist(),average='binary',zero_division = 0)               
    recall = recall_score(true.cpu().tolist(), pred.cpu().tolist(),average='binary',zero_division = 0)
    f1 = f1_score(true.cpu().tolist(), pred.cpu().tolist(),average='binary',zero_division = 0)                        
    accuracy = accuracy_score(true.cpu().tolist(), pred.cpu().tolist())
    if est_AUC :
        AUC =  roc_auc_score(true.cpu().tolist(), pred_cls.cpu().tolist()) 
        return [precision, recall, accuracy, f1, AUC]    
    else:
        return [precision, recall, accuracy, f1]

def estimate_metric_binary_con_eval(pred_cls, true_cls, pred_prob=None):
    true = (torch.eq(true_cls.cuda(), pred_cls.cuda())).long()
    binary_AUC = roc_auc_score(true.cpu().tolist(), pred_prob.cpu().tolist())
    return binary_AUC

'''
(type_3) low_level: precision, recall, AUC 
'''
def check_precision(pred_cls, true_cls, pred_prob=None, thres=None):
    pred = pred_prob.ge(thres).long() #[1,0,,,,]
    true = (torch.eq(true_cls.cuda(), pred_cls)).long() #[1,1,...]
    pred_true = pred * true
    # pred_true_index = pred.long().nonzero(as_tuple=True)[0] #[]
    true_pred_cls = (torch.eq(torch.index_select(true_cls.cuda(), 0, pred_true_index), torch.index_select(max_idx_w, 0, pred_true_index))).long()
    true_pred = torch.index_select(true, 0, pred_true_index)
    con_precision = torch.count_nonzero(true_pred).sum() / torch.count_nonzero(true_pred_cls).sum()
    print(con_precision)   




class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child

    """
    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open('/dev/stdin')
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin

class FeatureL2Norm(nn.Module):
    """
    Implementation by Ignacio Rocco
    paper: https://arxiv.org/abs/1703.05593
    project: https://github.com/ignacio-rocco/cnngeometric_pytorch
    """
    def __init__(self):
        super(FeatureL2Norm, self).__init__()

    def forward(self, feature, dim=1):
        epsilon = 1e-6
        norm = torch.pow(torch.sum(torch.pow(feature, 2), dim) + epsilon, 0.5).unsqueeze(dim).expand_as(feature)
        return torch.div(feature, norm)


class Con_estimator(nn.Module):
    def __init__(self, num_class, con_net, con_net_with_thres, epsilon, topk, in_features = 128, iteration = 0, warmup_max_iteration = 0, is_detach_con_ETE = None, is_negative_pair = None):
        super().__init__()
        self.con_net = con_net
        self.con_net_with_thres = con_net_with_thres
        self.l2norm = FeatureL2Norm()
        self.epsilon = epsilon
        self.topk = topk
        self.current_iteration = iteration
        self.warmup_max_iteration = warmup_max_iteration     
        self.is_detach_con_ETE = is_detach_con_ETE
        self.is_negative_pair = is_negative_pair    

        self.proj_feat = nn.Sequential(nn.Linear(128, 64),
                                        nn.ReLU())                                                
        self.proj_cls = nn.Sequential(nn.Linear(num_class, 64),
                                        nn.ReLU())
        self.con_estimator = nn.Sequential(nn.Linear(128, 64), 
                                            nn.ReLU(), 
                                            nn.Linear(64,32),
                                            nn.ReLU()) # sigmoid
        self.last_layer = nn.Linear(32, 1)
        
                               


    def forward(self, x):
        device = torch.cuda.current_device()
        
        feat, logits = x[0].detach(), x[1].detach()
        #feat: L2norm, logit: softmax
        feat = self.l2norm(feat)
        class_prob = torch.softmax(logits, dim=-1)
        if self.topk:
            bz, cls_num = class_prob.shape
            topk_index = torch.topk(class_prob, k=self.topk, dim=1)[1]
            topk_val = torch.topk(class_prob, k=self.topk, dim=1)[0]

        projected_feat = self.proj_feat(feat)
        projected_class_prob = self.proj_cls(topk_val)
        est_input = torch.cat([projected_feat, projected_class_prob], dim=1)

        con_output = self.con_estimator(est_input)
        con_output = self.last_layer(con_output)
        con_output = torch.sigmoid(con_output)
 
        return con_output.squeeze(1)



class Con_estimator_2(nn.Module):
    def __init__(self, num_class, con_net, con_net_with_thres, epsilon, topk, in_features = 128, iteration = 0, warmup_max_iteration = 0, is_detach_con_ETE = None, is_negative_pair = None):
        super().__init__()
        self.con_net = con_net
        self.con_net_with_thres = con_net_with_thres
        self.l2norm = FeatureL2Norm()
        self.epsilon = epsilon
        self.topk = topk
        self.current_iteration = iteration
        self.warmup_max_iteration = warmup_max_iteration     
        self.is_detach_con_ETE = is_detach_con_ETE
        self.is_negative_pair = is_negative_pair    

        self.proj_feat = nn.Sequential(nn.Linear(128, 64),
                                        nn.ReLU())                                                
        self.proj_cls = nn.Sequential(nn.Linear(self.topk, 64),
                                        nn.ReLU())
        self.con_estimator = nn.Sequential(nn.Linear(128, 64), 
                                            nn.ReLU(), 
                                            nn.Linear(64,32),
                                            nn.ReLU()) # sigmoid
        self.last_layer = nn.Linear(32, 1)
            
        torch.nn.init.zeros_(self.last_layer.weight)
        torch.nn.init.zeros_(self.last_layer.bias)
            
    def forward(self, x):
        device = torch.cuda.current_device()
        
        feat, logits = x[0].detach(), x[1].detach()
        #feat: L2norm, logit: softmax
        feat = self.l2norm(feat)
        class_prob = torch.softmax(logits, dim=-1)
        if self.topk:
            bz, cls_num = class_prob.shape
            topk_index = torch.topk(class_prob, k=self.topk, dim=1)[1]
            topk_val = torch.topk(class_prob, k=self.topk, dim=1)[0]

        projected_feat = self.proj_feat(feat)
        projected_class_prob = self.proj_cls(topk_val)
        est_input = torch.cat([projected_feat, projected_class_prob], dim=1)

        con_output = self.con_estimator(est_input)
        con_output = self.last_layer(con_output)
        con_output = torch.sigmoid(con_output)
 
        return con_output.squeeze(1)
            
class ConMatch2:
    def __init__(self, net_builder, con_net, num_classes, ema_m, T, p_cutoff, p_cutoff_con, lambda_u, lambda_p, lambda_c, \
                con_net_with_thres = False, epsilon = -10, con_net_with_softmax = False, topk= None, hard_label=True, \
                warmup_max_iteration = 0, is_detach_unsup_ETE = True , is_detach_con_ETE = True, is_negative_pair = False, 
                t_fn=None, p_fn=None, it=0,  num_eval_iter=1000, tb_log=None, logger=None):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """

        super(ConMatch2, self).__init__()
        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py

        self.model = net_builder(num_classes=num_classes)
        self.ema_model = None
        self.ema_model_con = None
        #create confidence estimator
        self.con_net = con_net
        self.con_net_with_softmax = con_net_with_softmax
        self.con_net_with_thres = con_net_with_thres
        self.it = 0
        self.epsilon = epsilon
        self.con_estimator = Con_estimator(num_class = num_classes,
                                            con_net = con_net,
                                            con_net_with_thres= con_net_with_thres,
                                            epsilon = epsilon,
                                            topk = topk,
                                            iteration = self.it,
                                            warmup_max_iteration = warmup_max_iteration,
                                            is_detach_con_ETE = is_detach_con_ETE,
                                            is_negative_pair = is_negative_pair)
        self.bce_loss = nn.BCELoss(reduction='mean')        
        self.warmup_max_iteration = warmup_max_iteration
        self.is_detach_unsup_ETE = is_detach_unsup_ETE
                                           
        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T)  # temperature params function
        self.p_fn = Get_Scalar(p_cutoff)  # confidence cutoff function
        self.p_con_fn = Get_Scalar(p_cutoff_con)
        self.lambda_u = lambda_u
        self.lambda_p = lambda_p
        self.lambda_c = lambda_c
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        self.compare_con = False
        self.optimizer = None
        self.scheduler = None
        self.optimizer_num = 0 
        self.lst = [[] for i in range(10)]
        self.abs_lst = [[] for i in range(10)]
        self.clsacc = [[] for i in range(10)]
        self.logger = logger
        self.print_fn = print if logger is None else logger.info

        self.bn_controller = Bn_Controller()

    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_dset(self, dset):
        self.ulb_dset = dset

    def set_optimizer(self, optimizer, scheduler=None):
        
        if isinstance(optimizer, tuple):
            self.optimizer = optimizer[0]
            self.scheduler = scheduler[0]
            
            self.optimizer_con = optimizer[1]
            self.scheduler_con = scheduler[1]
            self.optimizer_num = 2
        else:
            self.optimizer = optimizer
            self.scheduler = scheduler
            self.optimizer_num = 1
    def check_requires_grad(self, model):
        list_check = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                list_check.append(name)
        print(list_check)
        if list_check != []:
            print("requires_grad_false_task will be going")
            for name, param in model.named_parameters():
                for l in list_check:
                    if name == l:
                        param.requires_grad = False
        
        for param in model.parameters():
            assert param.requires_grad == False
    
    def train(self, args, logger=None):

        ngpus_per_node = torch.cuda.device_count()

        # EMA Init
        self.model.train()
        self.con_estimator.train()

        for param in self.model.parameters():
            param.requires_grad = True
        for param in self.con_estimator.parameters():
            param.requires_grad = True
        # (if) detach con_estimator
        # for param in self.con_estimator.parameters():
        #     param.requires_grad = False

        self.ema = EMA(self.model, self.ema_m)
        self.ema.register()

        self.ema_con = EMA(self.con_estimator, self.ema_m)
        self.ema_con.register()

        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)

        start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        best_eval_precision_con, best_precision_con_it = 0.0, 0
        best_eval_recall_con, best_recall_con_it = 0.0, 0
        best_eval_AUC_con, best_AUC_con_it = 0.0, 0
        best_eval_precision_fix95, best_precision_fix95_it = 0.0, 0
        best_eval_recall_fix95, best_recall_fix95_it = 0.0, 0 
        best_eval_AUC_fix_95, best_AUC_fix_95_it = 0.0, 0
        
        save_path = os.path.join(args.save_dir, args.save_name)                   
        with open(os.path.join(save_path, 'result_semi.txt'), 'w') as file:
            file.write(
            f'((best_eval_precision_con),(best_eval_recall_con),(best_eval_precision_fix95),(best_eval_recall_fix95),(best_eval_acc)\n') 
        
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext

        if args.resume:
            self.ema.load(self.ema_model)
            eval_dict = self.evaluate_con(args=args)
            print(eval_dict)
        if args.resume_con:
            self.ema_con.load(self.ema_model_con)
            eval_dict = self.evaluate_con(args=args)
            print(eval_dict)

        selected_label = torch.ones((len(self.ulb_dset),), dtype=torch.long, ) * -1
        selected_label = selected_label.cuda(args.gpu)
        classwise_acc = torch.zeros((args.num_classes,)).cuda(args.gpu)

        device = torch.cuda.current_device()
        for (_, x_lb, y_lb), (_, x_ulb_w, x_ulb_s1, x_ulb_s2, y_ulb) in zip(self.loader_dict['train_lb'],
                                                                  self.loader_dict['train_ulb']):        
            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()

            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s1.shape[0]

            x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2 = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s1.cuda(args.gpu), x_ulb_s2.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            y_logits_lb = []
            y_logits_ulb = []
            con_true = []
            con_pred = []
            pseudo_counter = Counter(selected_label.tolist())
            if max(pseudo_counter.values()) < len(self.ulb_dset):  # not all(5w) -1
                for i in range(args.num_classes):
                    classwise_acc[i] = pseudo_counter[i] / max(pseudo_counter.values())
            

            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s1, x_ulb_s2))
            ulb_last_index = num_lb + num_ulb*3

            feat, logits = self.model(inputs, is_tuple=True)
            if args.is_negative_pair:
                ulb_last_index = -num_lb
                y_lb_neg_pair_index = []
                y_lb_array = y_lb.cpu().detach().numpy()
                for i in range(num_lb):
                    y_lb_neg_cand = np.where(y_lb_array != y_lb_array[i])[0]
                    y_lb_neg_index = np.random.choice(y_lb_neg_cand, 1)
                    y_lb_neg_pair_index.append(np.asscalar(y_lb_neg_index))
                feat = torch.cat([feat, feat[0:num_lb]], dim=0)
                logits = torch.cat([logits, logits[y_lb_neg_pair_index]], dim=0)

            # hyper-params for update
            T = self.t_fn(self.it)
            p_cutoff = self.p_fn(self.it)            
            
            model_output = (feat, logits)
            con_output = self.con_estimator(model_output)
            #w/o nonlinear stretching
            # con_output = 1 / (1 + torch.exp(self.epsilon * (con_output_temp - p_cutoff)))

            con_loss = torch.tensor(0.0, device=device)
            logits_x_lb = logits[:num_lb]
            logits_x_ulb_w, logits_x_ulb_s1, logits_x_ulb_s2 = logits[num_lb:ulb_last_index].chunk(3)
            con_lb = con_output[:num_lb]
            con_x_ulb_w, con_x_ulb_s1, con_x_ulb_s2 = con_output[num_lb:ulb_last_index].chunk(3)
            #to_do_list: neg_term_with_weak_label
            # index_con_w = torch.ge(con_x_ulb_w, p_cutoff_con).nonzero(as_tuple =True)[0]
            # con_true_ul = torch.ones_like(index_con_w.size())
            # pdb.set_trace()
            sm_pred_w = F.softmax(logits_x_ulb_w, dim=-1)
            max_prob_w, max_idx_w = torch.max(sm_pred_w, dim=-1)
            con_true_w = torch.eq(max_idx_w, y_ulb.cuda()).float()
            #to compare real vs pred confidence
            if self.compare_con:
                sm_pred_w = F.softmax(logits_x_ulb_w, dim=-1)
                max_prob_w, max_idx_w = torch.max(sm_pred_w, dim=-1)
                con_true_w = torch.eq(max_idx_w, y_ulb.cuda()).float()
                right =  con_x_ulb_w * con_true_w
                right_mean = torch.sum(right)/torch.count_nonzero(right)
                flipped_con_true_w = custom_replace(con_true_w)
                wrong =  con_x_ulb_w * flipped_con_true_w
                wrong_mean = torch.sum(wrong)/torch.count_nonzero(wrong)
                pdb.set_trace()
            
            sm_pred = F.softmax(logits_x_lb, dim=-1)
            _, max_idx = torch.max(sm_pred, dim=-1)
            con_true_l = torch.eq(max_idx, y_lb).float()
            if args.is_negative_pair:
                con_lb_neg = con_output[-num_lb:]
                con_neg = torch.zeros_like(con_true_l)
                con_true = torch.cat([con_true_l, con_neg], dim=0)
                con_pred = torch.cat([con_lb, con_lb_neg], dim=0)
            else:
                con_true = con_true_l
                con_pred = con_lb
                # pdb.set_trace()
            # con_loss += confidence_loss(con_true, con_pred, torch.count_nonzero(con_true_l).sum(), num_lb, is_weighted_BCE = args.is_weighted_BCE)
            con_loss += self.bce_loss(con_pred, con_true)
            reg_con_loss_1 = torch.mean(torch.log(1/con_x_ulb_s1))
            loss_ours_1, _, _, _ = consistency_loss_con(logits_x_ulb_s1.detach(),
                                                                    logits_x_ulb_w,
                                                                    con_x_ulb_s1,
                                                                    'ce', T, p_cutoff,
                                                                    use_hard_labels=args.hard_label,
                                                                    use_threshold = False)            

            reg_con_loss_2 = torch.mean(torch.log(1/con_x_ulb_s2))
            loss_ours_2, _, _, _ = consistency_loss_con(logits_x_ulb_s2.detach(),
                                                                    logits_x_ulb_w,
                                                                    con_x_ulb_s2,
                                                                    'ce', T, p_cutoff,
                                                                    use_hard_labels=args.hard_label,
                                                                    use_threshold = False)   

            loss_ours_ss_1, mask, select, pseudo_lb = consistency_loss_con(logits_x_ulb_s2,
                                                                    logits_x_ulb_s1,
                                                                    con_x_ulb_s1.detach(),
                                                                    'ce', T, p_cutoff,
                                                                    use_hard_labels=args.hard_label,
                                                                    use_threshold = False) 
            
            loss_ours_ss_2, _, _, _ = consistency_loss_con(logits_x_ulb_s1,
                                                                    logits_x_ulb_s2,
                                                                    con_x_ulb_s2.detach(),
                                                                    'ce', T, p_cutoff,
                                                                    use_hard_labels=args.hard_label,
                                                                    use_threshold = False) 

            fixmatch_loss_1, mask, select, pseudo_lb = consistency_loss_fix(logits_x_ulb_s1,
                                                                    logits_x_ulb_w,
                                                                    'ce', T, 0.95,
                                                                    use_hard_labels=args.hard_label)


            fixmatch_loss_2, mask, select, pseudo_lb = consistency_loss_fix(logits_x_ulb_s2,
                                                                    logits_x_ulb_w,
                                                                    'ce', T, 0.95,
                                                                    use_hard_labels=args.hard_label)
                        
            sup_loss_fix = ce_loss(logits_x_lb, y_lb, reduction='mean')                       



            total_loss = sup_loss_fix + self.lambda_u * fixmatch_loss_1 + self.lambda_u * fixmatch_loss_2 \
                        + con_loss + args.reg_loss_ratio*(reg_con_loss_1 + reg_con_loss_2) \
                        + loss_ours_1 + loss_ours_2 + loss_ours_ss_1 + loss_ours_ss_2                        
            y_logits_lb.extend(logits_x_lb.tolist())
            y_logits_ulb.extend(logits_x_ulb_w.tolist())
            y_logits_lb = softmax(y_logits_lb,axis=-1)
            y_logits_ulb= softmax(y_logits_ulb,axis=-1)
            max_prob_l, max_idx_l = torch.max(torch.tensor(y_logits_lb).cuda(args.gpu),dim=-1)
            max_prob_w, max_idx_w = torch.max(torch.tensor(y_logits_ulb).cuda(args.gpu),dim=-1)
    
            lb_con_list = estimate_metric_binary_2(max_idx_l, y_lb, con_output[:num_lb], p_cutoff, est_AUC = False)        
            ulb_con_list = estimate_metric_binary_2(max_idx_w, y_ulb, con_output[num_lb: num_lb + num_ulb], p_cutoff, est_AUC = True)        


            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                if self.optimizer_num == 2:
                    scaler.step(self.optimizer)
                    scaler.step(self.optimizer_con)
                    self.scheduler.step()
                    self.scheduler_con.step()  
                elif self.optimizer_num == 1:
                    scaler.step(self.optimizer)
                    self.scheduler.step()              
                scaler.update()
            else:
                total_loss.backward()
                if (args.clip > 0):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), args.clip)
                self.optimizer.step()
                self.scheduler.step()

            self.ema.update()
            self.ema_con.update()
            self.model.zero_grad()
            self.con_estimator.zero_grad()
            end_run.record()
            torch.cuda.synchronize()

            # tensorboard_dict update
            tb_dict = {}
            y_lb_list = y_lb.cpu().tolist()
            y_ulb_list = y_ulb.cpu().tolist()

            max_idx_l_list = max_idx_l.cpu().tolist()
            max_idx_w_list = max_idx_w.cpu().tolist()

            if self.it % self.num_eval_iter == 0:
    
                # tb_dict['train/L_Precision_A'] = precision_score(y_lb_list, max_idx_l_list, average='macro')
                # tb_dict['train/L_Recall_A'] = recall_score(y_lb_list, max_idx_l_list, average='macro')
                # tb_dict['train/L_Acc_A'] = accuracy_score(y_lb_list, max_idx_l_list)
                # tb_dict['train/UL_Precision_A'] = precision_score(y_ulb_list, max_idx_w_list, average='macro')
                # tb_dict['train/UL_Recall_A'] = recall_score(y_ulb_list, max_idx_w_list, average='macro')
                
                # tb_dict['train/L_Precision'] = lb_con_list[0]
                # tb_dict['train/L_Recall'] = lb_con_list[1]
                # tb_dict['train/L_Acc'] = lb_con_list[2]
                # tb_dict['train/L_F1'] = lb_con_list[3]

                tb_dict['train/UL_Precision'] = ulb_con_list[0]
                tb_dict['train/UL_Recall'] = ulb_con_list[1]
                tb_dict['train/UL_Acc'] = ulb_con_list[2]
                tb_dict['train/UL_F1'] = ulb_con_list[3]
                tb_dict['train/UL_AUC'] = ulb_con_list[4]

                
                tb_dict['train/Con_pred_UL_ratio'] = torch.count_nonzero(con_true_w).sum()/ num_ulb
                tb_dict['train/Con_pred_L_ratio'] = torch.count_nonzero(con_true_l).sum()/ num_lb
                tb_dict['train/Con_pred_L_mean'] = torch.mean(con_lb).detach()
                tb_dict['train/Con_pred_UL_mean'] = torch.mean(con_x_ulb_w).detach()
                tb_dict['train/Con_pred_L_var'] = torch.var(con_lb).detach()
                tb_dict['train/Con_pred_UL_var'] = torch.var(con_x_ulb_w).detach()
                tb_dict['train/Con_pred_L_min'] = torch.min(con_lb).detach()
                tb_dict['train/Con_pred_UL_min'] = torch.min(con_x_ulb_w).detach()
                tb_dict['train/Con_pred_L_max'] = torch.max(con_lb).detach()
                tb_dict['train/Con_pred_UL_max'] = torch.max(con_x_ulb_w).detach()

                tb_dict['train/sup_loss'] = sup_loss_fix.detach()
                tb_dict['train/unsup_loss'] = (fixmatch_loss_1.detach() + fixmatch_loss_2.detach())
                tb_dict['train/loss_ours'] = (loss_ours_1.detach() + loss_ours_2.detach())
                tb_dict['train/loss_ours_ss'] = (loss_ours_ss_1.detach() + loss_ours_ss_2.detach())
                tb_dict['train/con_loss'] = con_loss.detach()
                tb_dict['train/reg_con_loss'] = (reg_con_loss_1.detach() + reg_con_loss_2.detach())
                tb_dict['train/total_loss'] = total_loss.detach()
                tb_dict['train/mask_ratio'] = 1.0 - mask.detach()
                tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
                tb_dict['lr_con'] = self.optimizer.param_groups[1]['lr']            
                tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch) / 1000.
                tb_dict['train/run_time'] = start_run.elapsed_time(end_run) / 1000.
            # Save model foptimzierdate(eval_dict)

            if self.it % self.num_eval_iter == 0:
                eval_dict = self.evaluate_con(args=args)
                self.model.train()

                tb_dict.update(eval_dict)

                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
               
                self.print_fn(
                    f"{self.it} iteration, USE_EMA: {self.ema_m != 0}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")

                if not args.multiprocessing_distributed or \
                        (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):

                    if self.it == best_it:
                        self.save_model('model_best.pth', save_path)

                    if not self.tb_log is None:
                        self.tb_log.update(tb_dict, self.it)
                


            self.it += 1
            self.con_estimator.current_iteration = self.it            
            del tb_dict
            start_batch.record()
            if self.it > 0.8 * args.num_train_iter:
                self.num_eval_iter = 1000

        eval_dict = self.evaluate(args=args)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict

    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None):
        self.model.eval()
        self.ema.apply_shadow()
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []
        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        self.ema.restore()
        self.model.train()
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC}

    @torch.no_grad()
    def evaluate_con(self, eval_loader=None, args=None, model_training = True):
        #for eval_model (open)
        self.model.eval()
        self.ema.apply_shadow()
        #for eval_con (open)        
        self.con_estimator.eval()
        self.ema_con.apply_shadow() 
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        total_loss = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []

        con_true = []
        con_pred = []

        max_prob_list = []
        max_idx_list = []

        p_cutoff_con = self.p_con_fn(self.it)
        p_cutoff = self.p_fn(self.it)

        for _, x, y in eval_loader:
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            if 'fusion' in self.con_net:
                feat, logits = self.model(x, is_tuple = True)
                model_output = (feat, logits)             
            
            loss = F.cross_entropy(logits, y, reduction='mean')
            logits_list = logits.clone().tolist()
            logits_list = softmax(logits_list,axis=-1)
            max_prob, max_idx = torch.max(torch.tensor(logits_list).cuda(args.gpu),dim=-1)
            max_prob_list.extend(max_prob.cpu().tolist())            
            max_idx_list.extend(max_idx.cpu().tolist())            
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            total_loss += loss.detach() * num_batch
            #conmatch_confidence

            con_true_val = torch.eq(y, max_idx)
            con_true.extend(con_true_val.cpu().tolist())
            
            con_output = self.con_estimator(model_output)
            con_pred.extend(con_output.cpu().tolist())                
            
            # con_true.extend(torch.eq(y, torch.max(torch.softmax(logits, dim=-1), dim=-1)[1]).cpu().tolist())
            # con_true.extend(torch.eq(y, max_idx).long().cpu().tolist())
            # con_pred.extend(con_output.ge(p_cutoff_con).long().cpu().tolist())
            # if self.con_net_with_thres:
            #     con_pred.extend(con_output.ge(p_cutoff).long().cpu().tolist())
            # else:
            #     con_pred.extend(con_output.ge(p_cutoff_con).long().cpu().tolist())

            #fixmatch_confidence
            # fix_pred.extend(max_prob.ge(p_cutoff_con).long().cpu().tolist())
        # print("(eval)", con_true.count(1), con_pred.count(1), con_pred.count(0))
        top1 = accuracy_score(y_true, y_pred)
        top5 = top_k_accuracy_score(y_true, y_logits, k=5)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        F1 = f1_score(y_true, y_pred, average='macro')
        AUC = roc_auc_score(y_true, y_logits, multi_class='ovo')

        max_prob_tensor = torch.tensor(max_prob_list).cuda(args.gpu)
        max_idx_tensor = torch.tensor(max_idx_list).cuda(args.gpu)
        y_true_tensor = torch.tensor(y_true).cuda(args.gpu)
        con_pred_tensor = torch.tensor(con_pred).cuda(args.gpu)
        
        AUC_con = estimate_metric_binary_con_eval(max_idx_tensor,y_true_tensor,con_pred_tensor)

        cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
        self.print_fn('confusion matrix:\n' + np.array_str(cf_mat))
        #for eval_model (close)        
        self.ema.restore() 
        self.model.train()
        #for eval_model (close)        
        self.ema_con.restore() 
        self.con_estimator.train()
    
        return {'eval/loss': total_loss / total_num, 'eval/top-1-acc': top1, 'eval/top-5-acc': top5,
                'eval/precision': precision, 'eval/recall': recall, 'eval/F1': F1, 'eval/AUC': AUC, 
                'eval/AUC_con': AUC_con,}

    #save : model + con_estimator
    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        self.model.eval()
        self.ema.apply_shadow()
        ema_model = self.model.state_dict()
        self.ema.restore()
        self.model.train()
        
        self.con_estimator.eval()
        self.ema_con.apply_shadow()
        ema_model_con = self.con_estimator.state_dict()
        self.ema_con.restore()
        self.con_estimator.train()

        torch.save({'model': self.model.state_dict(),
                    'con_estimator': self.con_estimator.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model,
                    'ema_model_con': ema_model_con},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    #save : con_estimator
    def save_model_con(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        # copy EMA parameters to ema_model for saving with model as temp
        
        # self.model.eval()
        # self.ema.apply_shadow()
        # ema_model = self.model.state_dict()
        # self.ema.restore()
        # (ema) con_estimator 
        self.con_estimator.eval()
        self.ema_con.apply_shadow()
        ema_model_con = self.con_estimator.state_dict()
        self.ema_con.restore()
        self.con_estimator.train()
        # dict key is same as model's pth
        torch.save({'model': self.con_estimator.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it,
                    'ema_model': ema_model_con},
                   save_filename)

        self.print_fn(f"model saved: {save_filename}")

    def load_model(self, load_path):
        device = torch.cuda.current_device()
        checkpoint = torch.load(load_path, map_location="cuda:{}".format(device))
        res = self.model.load_state_dict(checkpoint['model'], strict=True)
        print(res.missing_keys)
        # assert set(res.missing_keys) == set(["con_classifier.weight", "con_classifier.bias"])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'],strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model loaded for stage_1')
        
    def load_model_2(self, load_path):
        device = torch.cuda.current_device()
        checkpoint = torch.load(load_path, map_location="cuda:{}".format(device))
        res = self.model.load_state_dict(checkpoint['model'], strict=True)
        print(res.missing_keys)
        # assert set(res.missing_keys) == set(["con_classifier.weight", "con_classifier.bias"])
        self.ema_model = deepcopy(self.model)
        self.ema_model.load_state_dict(checkpoint['ema_model'],strict=True)      
        self.it = checkpoint['it']
        self.print_fn('model_loaded for stage_2')
        
    def load_model_3(self, load_path):
        device = torch.cuda.current_device()
        checkpoint = torch.load(load_path, map_location="cuda:{}".format(device))
     
        res = self.model.load_state_dict(checkpoint['model'], strict=True)
        self.ema_model = deepcopy(self.model)        
        self.ema_model.load_state_dict(checkpoint['ema_model'],strict=True)      
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

        res_con = self.con_estimator.load_state_dict(checkpoint['con_estimator'], strict=True)
        self.ema_model_con = deepcopy(self.con_estimator)        
        self.ema_model_con.load_state_dict(checkpoint['ema_model_con'],strict=True)      


        self.it = checkpoint['it']
        self.print_fn('model_loaded for stage_3')

    def load_model_con(self, load_path):
        device = torch.cuda.current_device()
        checkpoint = torch.load(load_path, map_location="cuda:{}".format(device))
        res = self.con_estimator.load_state_dict(checkpoint['con_estimator'], strict=True)
        print(res.missing_keys)
        # assert set(res.missing_keys) == set(["con_classifier.weight", "con_classifier.bias"])
        self.ema_model_con = deepcopy(self.con_estimator)
        self.ema_model_con.load_state_dict(checkpoint['con_estimator'],strict=True)        
        self.optimizer_con.load_state_dict(checkpoint['optimizer'])
        self.scheduler_con.load_state_dict(checkpoint['scheduler'])
        self.it = checkpoint['it']
        self.print_fn('model con loaded')

    def interleave_offsets(self, batch, nu):
        groups = [batch // (nu + 1)] * (nu + 1)
        for x in range(batch - sum(groups)):
            groups[-x - 1] += 1
        offsets = [0]
        for g in groups:
            offsets.append(offsets[-1] + g)
        assert offsets[-1] == batch
        return offsets

    def interleave(self, xy, batch):
        nu = len(xy) - 1
        offsets = self.interleave_offsets(batch, nu)
        xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
        for i in range(1, nu + 1):
            xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
        return [torch.cat(v, dim=0) for v in xy]


if __name__ == "__main__":
    pass
