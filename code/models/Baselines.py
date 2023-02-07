import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers import BertModel
from .layers import Attention


class bBbox(torch.nn.Module):
    def __init__(self,fea_dim):
        super(bBbox, self).__init__()
        self.img_dim = 4096
        self.attention1 = Attention(dim=128,heads=4)
        self.attention2 = Attention(dim=128,heads=4)

        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim),torch.nn.ReLU())

        self.classifier = nn.Linear(fea_dim,2)
    
    def forward(self,  **kwargs):
        frames=kwargs['bbox_vgg']
        fea_img = self.linear_img(frames) 
        fea_img = torch.reshape(fea_img, (-1, 45, 128)) 
        fea_img = self.attention1(fea_img)
        fea_img = torch.mean(fea_img, -2) 
        fea_img = torch.reshape(fea_img, (-1, 83, 128))
        fea_img = self.attention2(fea_img)
        fea_img = torch.mean(fea_img, -2) 
        output = self.classifier(fea_img)
        return output, fea_img

class bC3D(torch.nn.Module):
    def __init__(self,fea_dim):
        super(bC3D, self).__init__()
        # self.video_dim = 4096
        self.video_dim = 2048
        self.attention = Attention(dim=128,heads=4)

        self.linear_video = nn.Sequential(torch.nn.Linear(self.video_dim, fea_dim),torch.nn.ReLU())

        self.classifier = nn.Linear(fea_dim,2)
    
    def forward(self,  **kwargs):
        c3d = kwargs['c3d'] 
        fea_video = self.linear_video(c3d)
        fea_video = self.attention(fea_video)
        fea_video = torch.mean(fea_video, -2)
        output = self.classifier(fea_video)
        return output

class bVGG(torch.nn.Module):
    def __init__(self,fea_dim):
        super(bVGG, self).__init__()
        # self.img_dim = 4096
        self.img_dim = 2048
        self.attention = Attention(dim=128,heads=4)

        self.linear_img = nn.Sequential(torch.nn.Linear(self.img_dim, fea_dim),torch.nn.ReLU())

        self.classifier = nn.Linear(fea_dim,2)
    
    def forward(self,  **kwargs):
        frames=kwargs['frames'] 
        fea_img = self.linear_img(frames) 
        fea_img = self.attention(fea_img)
        fea_img = torch.mean(fea_img, -2)
        output = self.classifier(fea_img)
        return output

class bVggish(torch.nn.Module):
    def __init__(self,fea_dim):
        super(bVggish, self).__init__()
        # self.audio_dim = 128
        self.attention = Attention(dim=128,heads=4)

        self.vggish_layer = torch.hub.load('./torchvggish/', 'vggish', source = 'local')        
        net_structure = list(self.vggish_layer.children())      
        self.vggish_modified = nn.Sequential(*net_structure[-2:-1])

        self.classifier = nn.Linear(fea_dim,2)
    
    def forward(self,  **kwargs):
        audioframes=kwargs['audioframes']
        fea_audio = self.vggish_modified(audioframes)
        fea_audio = self.attention(fea_audio)
        fea_audio = torch.mean(fea_audio, -2)
        print (fea_audio.shape)
        output = self.classifier(fea_audio)
        return output, fea_audio


class bBert(torch.nn.Module):
    def __init__(self,bert_model,fea_dim, dropout):
        super(bBert, self).__init__()
        self.text_dim = 768

        self.bert = BertModel.from_pretrained(bert_model).requires_grad_(False)

        self.linear_text = nn.Sequential(torch.nn.Linear(self.text_dim, fea_dim),torch.nn.ReLU())
        self.classifier = nn.Linear(fea_dim,2)
    
    def forward(self,  **kwargs):
        title_inputid = kwargs['title_inputid']
        title_mask=kwargs['title_mask']
        fea_text=self.bert(title_inputid,attention_mask=title_mask)[1]
        fea_text=self.linear_text(fea_text) 
        output = self.classifier(fea_text)
        return output,fea_text

class bTextCNN(nn.Module):
    def __init__(self, fea_dim, vocab_size):
        super(bTextCNN, self).__init__()
        self.vocab_size = vocab_size
        self.fea_dim=fea_dim

        self.channel_in = 1
        self.filter_num = 14
        self.window_size = [3,4,5]

        self.textcnn =nn.ModuleList([nn.Conv2d(self.channel_in, self.filter_num, (K,self.vocab_size)) for K in self.window_size])
        self.linear = nn.Sequential(torch.nn.Linear(len(self.window_size) * self.filter_num, self.fea_dim),torch.nn.ReLU())
        self.classifier = nn.Linear(self.fea_dim,2)

    def forward(self, **kwargs):
        title_w2v = kwargs['title_w2v'] 
        text = title_w2v.unsqueeze(1)
        text = [F.relu(conv(text)).squeeze(3) for conv in self.textcnn]
        text = [F.max_pool1d(i.squeeze(2), i.shape[-1]).squeeze(2) for i in text]
        fea_text = torch.cat(text, 1)
        fea_text = self.linear(fea_text) 

        output = self.classifier(fea_text)
        
        return output

class bComments(torch.nn.Module):
    def __init__(self,bert_model,fea_dim):
        super(bComments, self).__init__()
        self.comment_dim = 768
        self.bert = BertModel.from_pretrained(bert_model).requires_grad_(False)
        self.attention = Attention(dim=128,heads=4)
        self.linear_comment = nn.Sequential(torch.nn.Linear(self.comment_dim, fea_dim),torch.nn.ReLU())
        self.classifier = nn.Linear(fea_dim,2)
    
    def forward(self,  **kwargs):
        comments_inputid = kwargs['comments_inputid']
        comments_mask=kwargs['comments_mask']
        comments_feature=[]
        for i in range(comments_inputid.shape[0]):
            bert_fea=self.bert(comments_inputid[i], attention_mask=comments_mask[i])[1]
            comments_feature.append(bert_fea)
        comments_feature=torch.stack(comments_feature)
        fea_comments=self.linear_comment(comments_feature)
        print (fea_comments.shape)
        fea_comments = self.attention(fea_comments)
        fea_comments = torch.mean(fea_comments, -2)
        output = self.classifier(fea_comments)
        return output


