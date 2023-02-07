

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from sklearn.metrics import *
from transformers import BertModel
from zmq import device

from .layers import *


class TextCNN(nn.Module):
    def __init__(self, fea_dim, vocab_size):
        super(TextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.fea_dim=fea_dim

        self.channel_in = 1
        self.filter_num = 14
        self.window_size = [3,4,5]

        self.textcnn =nn.ModuleList([nn.Conv2d(self.channel_in, self.filter_num, (K,self.vocab_size)) for K in self.window_size])
        self.linear = nn.Sequential(torch.nn.Linear(len(self.window_size) * self.filter_num, self.fea_dim),torch.nn.ReLU())

    def forward(self, inputs):
        text = inputs.unsqueeze(1)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.textcnn]
        text = [F.max_pool1d(i.squeeze(2), i.shape[-1]).squeeze(2) for i in text]
        fea_text = torch.cat(text, 1)
        fea_text = self.linear(fea_text) 
        
        return fea_text


class VideoEncoder(nn.Module):
    def __init__(self,emb_dim,fea_dim):
        super(VideoEncoder, self).__init__()

        self.emb_dim = emb_dim
        self.linear1 = torch.nn.Linear(self.emb_dim, self.emb_dim, bias=False)
        self.linear2 = nn.Sequential(torch.nn.Linear(self.emb_dim, fea_dim),torch.nn.ReLU())

    def forward(self, input_thumb, input_L):
        input_ALL = torch.cat((input_L, input_thumb),1) #（bs,len+1,4096）
        fea_A = torch.bmm(input_thumb,self.linear1(input_ALL).permute(0,2,1)) # (bs, 1, len+1)
        fea_alpha = F.softmax(fea_A) # (bs, 1, len+1)
        fea_V = torch.matmul(fea_alpha,input_ALL).squeeze() # (bs, 4096)
        fea = self.linear2(fea_V)
        return fea

class ReverseLayerF(Function):
    #@staticmethod
    def forward(self, x):
        self.lambd = 1
        return x.view_as(x)

    #@staticmethod
    def backward(self, grad_output):
        return (grad_output * -self.lambd)

def grad_reverse(x):
    return ReverseLayerF.apply(x)


class FANVMModel(torch.nn.Module):
    def __init__(self,bert_model,fea_dim):
        super(FANVMModel, self).__init__()
        self.text_dim = 768
        self.img_dim = 4096
        self.topic_dim = 15

        self.bert = BertModel.from_pretrained(bert_model).requires_grad_(False)
        self.title_encoder = TextCNN(fea_dim, self.text_dim)
        self.comments_encoder = BiLSTM(self.text_dim,300,fea_dim)
        self.video_encoder = VideoEncoder(self.img_dim,fea_dim)

        self.gate_m1 = torch.nn.Linear(fea_dim*2,1)
        self.gate_m2 = torch.nn.Linear(fea_dim*2,1)

        self.classifier = nn.Linear(fea_dim*2,2)
        self.classifier_topic = nn.Linear(fea_dim*3,self.topic_dim)
    
    def forward(self, **kwargs):
        title_inputid = kwargs['title_inputid']#(batch,512)
        title_mask = kwargs['title_mask']#(batch,512)
        fea_text = self.bert(title_inputid,attention_mask=title_mask)[0]  #(bs,seq,768)
        fea_text = self.title_encoder(fea_text)
        fea_R = fea_text # (bs, 128)

        comments_inputid = kwargs['comments_inputid']#(batch,20,250)
        comments_mask=kwargs['comments_mask']#(batch,20,250)
        comments_like=kwargs['comments_like']
        comments_feature=[]
        for i in range(comments_inputid.shape[0]):
            bert_fea=self.bert(comments_inputid[i], attention_mask=comments_mask[i])[0]
            comments_feature.append(self.comments_encoder(bert_fea))
        comments_feature=torch.stack(comments_feature) #(batch,seq,fea_dim)
        fea_comments =[]
        for v in range(comments_like.shape[0]): # batch内循环
            # print (reviews_like[v])
            comments_weight=torch.stack([torch.true_divide((i+1),(comments_like[v].shape[0]+comments_like[v].sum())) for i in comments_like[v]])
            comments_fea_reweight = torch.sum(comments_feature[v]*(comments_weight.reshape(comments_weight.shape[0],1)),dim=0)
            fea_comments.append(comments_fea_reweight)
        fea_comments = torch.stack(fea_comments)
        fea_H = fea_comments # (bs, 600)

        frames = kwargs['frames'] # (bs, 30, 4096)
        frame_thumb = kwargs['frame_thmub'] # (bs,1,4096)
        fea_video = self.video_encoder(frame_thumb, frames)
        fea_V = fea_video # (bs, 128)

        s = kwargs['s']
        
        ## fusion: title, frames 
        m1 = self.gate_m1(torch.cat((fea_V, fea_R),1))
        fea_P = torch.add(torch.mul(m1,fea_V),torch.mul((1-m1),fea_R)) 
        ## fusion: comments, title
        m2 = s.reshape((s.shape[0],1))
        fea_E = torch.add(torch.mul(fea_H,m2),torch.mul(fea_R,(1-m2)))

        fea_fnd = torch.cat((fea_P,fea_E),1).to(torch.float32)
        output = self.classifier(fea_fnd)

        fea_topic = torch.cat((fea_H, fea_R, fea_V),1)
        fea_reverse = grad_reverse(fea_topic)
        output_topic = self.classifier_topic(fea_reverse)

        return output,output_topic,fea_fnd
