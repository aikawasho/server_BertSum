# -*- coding: utf-8 -*-
#モデルの定義

import torch
from pytorch_pretrained_bert import BertConfig
from .models.model_builder import Summarizer
from .prepro.data_builder import _format_to_bert_pred
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
# パラメータ
class bert_args():
    def __init__(self,min_src_ntokens = 5, max_src_ntokens = 200, min_nsents = 3, max_nsents = 1000, bert_config_path = None,temp_dir = None,dropout=0.1 ,encoder = 'classifier',param_init = 0,param_init_glorot = True, ff_size = 2048,inter_layers=2,heads=8,rnn_size=768):
        self.min_src_ntokens = min_src_ntokens
        self.max_src_ntokens = max_src_ntokens
        self.min_nsents = min_nsents
        self.max_nsents = max_nsents
        self.temp_dir = '../temp'
        self.bert_config_path = './BertSum/models/Japanese_L-12_H-768_A-12_E-30_BPE_WWM_transformers/config.json'
        self.encoder = encoder
        self.param_init = param_init 
        self.param_init_glorot = param_init_glorot
        self.ff_size = ff_size
        self.inter_layers = inter_layers
        self.heads = heads
        self.dropout = dropout
        self.rnn_size = rnn_size
        
args = bert_args()

test_from  = './BertSum/models/model_step_2000.pt'
checkpoint = torch.load(test_from, map_location=torch.device('cpu'))
config = BertConfig.from_json_file(args.bert_config_path)
model = Summarizer(args, device="cpu", load_pretrained_bert=False, bert_config = config)
model.load_cp(checkpoint)

#一連の流れ関数化
def Bertsum_pred(src):
    _pred = ''
    b_data_dict = _format_to_bert_pred(args,src)


    pre_src = b_data_dict['src']
    pre_segs = b_data_dict['segs']
    pre_clss = b_data_dict['clss']
    src = torch.tensor([pre_src])
    segs = torch.tensor([pre_segs])
    mask = ~(src == 0)

    clss = torch.tensor([pre_clss])
    mask_cls = ~(clss == -1)
    clss[clss == -1] = 0
    sent_scores, mask = model(src, segs, clss, mask, mask_cls)
    sent_scores = sent_scores + mask.float()
    sent_scores = sent_scores.cpu().data.numpy()
    
    selected_ids = np.argsort(-sent_scores, 1)
    selected_ids = list(flatten_2d(selected_ids))
    src_str = b_data_dict['src_txt']
    _pred = []
    scores = sent_scores[0]
    mean = sum(scores)/len(scores)
    scores = (scores-mean)/np.var(scores)
    print(scores)

    for i, idx in enumerate(selected_ids):
        #print(idx)
        #if(len(src_str[i])==0):
            #continue

        candidate = src_str[idx].replace(' ','')  
        if scores[idx] > 40:
            _pred.append(candidate)


    return _pred

#平滑化
def flatten_2d(data):
    for block in data:
        for elem in block:
            yield elem
