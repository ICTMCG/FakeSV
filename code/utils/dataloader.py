

import os
import pickle

import h5py
import jieba
import jieba.analyse as analyse
import numpy as np
import pandas as pd
import torch
from scipy.spatial import distance
from sklearn import preprocessing
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from torch.utils.data import Dataset
from transformers import BertTokenizer

def str2num(str_x):
    if isinstance(str_x, float):
        return str_x
    elif str_x.isdigit():
        return int(str_x)
    elif 'w' in str_x:
        return float(str_x[:-1])*10000
    elif '亿' in str_x:
        return float(str_x[:-1])*100000000
    else:
        print ("error")
        print (str_x)
        

class SVFENDDataset(Dataset):

    def __init__(self, path_vid, datamode='title+ocr'):
        
        with open('./data/dict_vid_audioconvfea.pkl', "rb") as fr:
            self.dict_vid_convfea = pickle.load(fr)

        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)
        self.data_complete = self.data_complete[self.data_complete['label']!=2] # label: 0-real, 1-fake, 2-debunk

        self.framefeapath='./data/ptvgg19_frames/'
        self.c3dfeapath='./data/c3d/'

        self.vid = []
        
        with open('./data/vids/'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)    
        self.data.reset_index(inplace=True)  

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        self.datamode = datamode
        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        # label 
        label = 0 if item['annotation']=='真' else 1
        label = torch.tensor(label)

        # text
        if self.datamode == 'title+ocr':
            title_tokens = self.tokenizer(item['description']+' '+item['ocr'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'ocr':
            title_tokens = self.tokenizer(item['ocr'], max_length=512, padding='max_length', truncation=True)
        elif self.datamode == 'title':
            title_tokens = self.tokenizer(item['description'], max_length=512, padding='max_length', truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        # comments
        comments_inputid = []
        comments_mask = []
        for comment in item['comments']:
            comment_tokens = self.tokenizer(comment, max_length=250, padding='max_length', truncation=True)
            comments_inputid.append(comment_tokens['input_ids'])
            comments_mask.append(comment_tokens['attention_mask'])
        comments_inputid = torch.LongTensor(np.array(comments_inputid)) 
        comments_mask = torch.LongTensor(np.array(comments_mask))
        
        comments_like = []
        for num in item['comments_like']:
            num_like = num.split(" ")[0] 
            comments_like.append(str2num(num_like))
        comments_like = torch.tensor(comments_like)
        
        # audio
        audioframes = self.dict_vid_convfea[vid]
        audioframes = torch.FloatTensor(audioframes)
        
        # frames
        frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)
        
        # video
        c3d = h5py.File(self.c3dfeapath+vid+".hdf5", "r")[vid]['c3d_features']
        c3d = torch.FloatTensor(c3d)

        # # user
        try: 
            if item['is_official'] == 1:
                intro = "个人认证"
            elif item['is_official'] == 2:
                intro = "机构认证"
            elif item['is_official'] == 0:
                intro = "未认证"
            else: 
                intro = "认证状态未知"
        except: 
            intro = "认证状态未知"

        for key in ['poster_intro', 'content_verify']: 
            try:
                intro = intro + '   ' + item[key]
            except:
                intro += '  '
        intro_tokens = self.tokenizer(intro, max_length=50, padding='max_length', truncation=True)
        intro_inputid = torch.LongTensor(intro_tokens['input_ids'])
        intro_mask = torch.LongTensor(intro_tokens['attention_mask'])

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'audioframes': audioframes,
            'frames':frames,
            'c3d': c3d,
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'comments_like': comments_like,
            'intro_inputid': intro_inputid,
            'intro_mask': intro_mask,
        }


def split_word(df):
    title = df['description'].values
    comments = df['comments'].apply(lambda x:' '.join(x)).values
    text = np.concatenate([title, comments],axis=0)
    analyse.set_stop_words('./data/stopwords.txt')
    all_word = [analyse.extract_tags(txt) for txt in text.tolist()]
    corpus = [' '.join(word) for word in all_word]
    return corpus


class FANVMDataset_train(Dataset):

    def __init__(self, path_vid_train):
        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)

        self.framefeapath='./data/ptvgg19_frames/'
        self.thumbframefeapath='./data/ptvgg19_frame_thumb/'

        self.vid_train = []
        with open('./data/vids/'+path_vid_train, "r") as fr:
            for line in fr.readlines():
                self.vid_train.append(line.strip())
        self.data_train = self.data_complete[self.data_complete.video_id.isin(self.vid_train)]  
        self.data_train['video_id'] = self.data_train['video_id'].astype('category')
        self.data_train['video_id'].cat.set_categories(self.vid_train, inplace=True)
        self.data_train.sort_values('video_id', ascending=True, inplace=True)    
        self.data_train.reset_index(inplace=True)    

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        corpus = split_word(self.data_train)
        tfidf = TfidfVectorizer().fit_transform(corpus)
        lda = LatentDirichletAllocation(n_components=15,random_state=2022)
        docres = lda.fit_transform(tfidf)
        self.topic_title = []
        s = []
        for idx in range(self.data_train.shape[0]):
            theta_title = docres[idx]
            self.topic_title.append(theta_title)
            theta_comments = docres[idx+self.data_train.shape[0]]
            s.append(distance.jensenshannon(theta_title, theta_comments) ** 2)
        min_max_scaler = preprocessing.MinMaxScaler()
        s_minMax = min_max_scaler.fit_transform(np.array(s).reshape(-1, 1))
        self.s_minMax = s_minMax.reshape(s_minMax.shape[0])

    def __len__(self):
        return self.data_train.shape[0]
     
    def __getitem__(self, idx):
        item = self.data_train.iloc[idx]
        vid = item['video_id']

        label = 1 if item['annotation']=='假' else 0
        label = torch.tensor(label)
        
        title_tokens = self.tokenizer(item['description'], max_length=512, padding='max_length', truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        comments_inputid = []
        comments_mask = []
        for comment in item['comments']:
            comment_tokens = self.tokenizer(comment, max_length=250, padding='max_length', truncation=True)
            comments_inputid.append(comment_tokens['input_ids'])
            comments_mask.append(comment_tokens['attention_mask'])
        comments_inputid = torch.LongTensor(comments_inputid) 
        comments_mask = torch.LongTensor(comments_mask)
        
        comments_like = []
        for num in item['comments_like']:
            num_like = num.split(" ")[0] 
            comments_like.append(str2num(num_like))
        comments_like = torch.tensor(comments_like)
        
        frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)

        frame_thmub = pickle.load(open(os.path.join(self.thumbframefeapath,vid+'.pkl'),'rb'))
        frame_thmub = torch.FloatTensor(frame_thmub)

        s = self.s_minMax[idx]
        s = torch.tensor(s)

        topic_title = self.topic_title[idx]
        topic_title = torch.FloatTensor(topic_title)
        

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'comments_like': comments_like,
            'frames':frames,
            'frame_thmub':frame_thmub,
            's':s,
            'label_event':topic_title,
        }


class FANVMDataset_test(Dataset):

    def __init__(self, path_vid_train, path_vid_test):
        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)

        self.framefeapath='./data/ptvgg19_frames/'
        self.thumbframefeapath='./data/ptvgg19_frame_thumb/'

        self.vid_train = []
        with open('./data/vids/'+path_vid_train, "r") as fr:
            for line in fr.readlines():
                self.vid_train.append(line.strip())
        self.data_train = self.data_complete[self.data_complete.video_id.isin(self.vid_train)]  
        self.data_train['video_id'] = self.data_train['video_id'].astype('category')
        self.data_train['video_id'].cat.set_categories(self.vid_train, inplace=True)
        self.data_train.sort_values('video_id', ascending=True, inplace=True)    
        self.data_train.reset_index(inplace=True)  

        self.vid_test = []
        with open('./data/vids/'+path_vid_test, "r") as fr:
            for line in fr.readlines():
                self.vid_test.append(line.strip())        
        self.data_test = self.data_complete[self.data_complete.video_id.isin(self.vid_test)]  
        self.data_test['video_id'] = self.data_test['video_id'].astype('category')
        self.data_test['video_id'].cat.set_categories(self.vid_test, inplace=True)
        self.data_test.sort_values('video_id', ascending=True, inplace=True)    
        self.data_test.reset_index(inplace=True)  

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        # Use data_train to train
        corpus_train = split_word(self.data_train)
        tfidf = TfidfVectorizer()
        tfidf_matrix_train = tfidf.fit_transform(corpus_train)
        lda = LatentDirichletAllocation(n_components=15,random_state=2022)
        lda.fit(tfidf_matrix_train)

        # apply on data_test
        corpus_test = split_word(self.data_test)
        tfidf_matrix_test = tfidf.transform(corpus_test)
        docres = lda.transform(tfidf_matrix_test)

        s = []
        self.topic_title = []
        for idx in range(self.data_test.shape[0]):
            theta_title = docres[idx]
            self.topic_title.append(theta_title)
            theta_comments = docres[idx+self.data_test.shape[0]]
            s.append(distance.jensenshannon(theta_title, theta_comments) ** 2)
        min_max_scaler = preprocessing.MinMaxScaler()
        s_minMax = min_max_scaler.fit_transform(np.array(s).reshape(-1, 1))
        self.s_minMax = s_minMax.reshape(s_minMax.shape[0])

    def __len__(self):
        return self.data_test.shape[0]
     
    def __getitem__(self, idx):
        item = self.data_test.iloc[idx]
        vid = item['video_id']

        label = 1 if item['annotation']=='假' else 0
        label = torch.tensor(label)
        
        title_tokens = self.tokenizer(item['description'], max_length=512, padding='max_length', truncation=True)
        title_inputid = torch.LongTensor(title_tokens['input_ids'])
        title_mask = torch.LongTensor(title_tokens['attention_mask'])

        comments_inputid = []
        comments_mask = []
        for comment in item['comments']:
            comment_tokens = self.tokenizer(comment, max_length=250, padding='max_length', truncation=True)
            comments_inputid.append(comment_tokens['input_ids'])
            comments_mask.append(comment_tokens['attention_mask'])
        comments_inputid = torch.LongTensor(comments_inputid) 
        comments_mask = torch.LongTensor(comments_mask)
        
        comments_like = []
        for num in item['comments_like']:
            num_like = num.split(" ")[0] 
            comments_like.append(str2num(num_like))
        comments_like = torch.tensor(comments_like)
        
        frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)

        frame_thmub = pickle.load(open(os.path.join(self.thumbframefeapath,vid+'.pkl'),'rb'))
        frame_thmub = torch.FloatTensor(frame_thmub)

        s = self.s_minMax[idx]
        s = torch.tensor(s)

        topic_title = self.topic_title[idx]
        topic_title = torch.FloatTensor(topic_title)
        

        return {
            'label': label,
            'title_inputid': title_inputid,
            'title_mask': title_mask,
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'comments_like': comments_like,
            'frames':frames,
            'frame_thmub':frame_thmub,
            's':s,
            'label_event':topic_title,
        }


class TikTecDataset(Dataset):

    def __init__(self, path_vid):
        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)

        self.vid = []
        with open(f'./data/vids/{path_vid}', "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_complete[self.data_complete['video_id'].isin(self.vid)]

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        label = 1 if item['label']=='假' else 0
        label = torch.tensor(label)

        max_K = 200  # max num of frames
        max_N = 500  # max num of ASR words

        # get caption feature
        with open('./data/caption_w2v_pad%s.pkl' % vid, 'rb') as f:
            caption_feature = pickle.load(f)  # (num_frame, 100, 300)
        if max_K / caption_feature.shape[0] >= 2:
            times = math.floor(max_K / caption_feature.shape[0])
            caption_feature = caption_feature.repeat_interleave(times, dim=0)
        elif caption_feature.shape[0] > max_K:
            times = math.ceil(caption_feature.shape[0] / max_K)
            caption_feature = caption_feature[::times][:max_K]
        actual_K = caption_feature.shape[0]
        caption_feature = torch.cat([caption_feature, torch.zeros((max_K - caption_feature.shape[0], 100, 300))], dim=0)

        # get visual feature
        with open( './data/vgg19_result%s.pkl' % vid, 'rb') as f:
            visual_feature = pickle.load(f)  # (num_frame, 45, 1000)
        if max_K / visual_feature.shape[0] >= 2:
            times = math.floor(max_K / visual_feature.shape[0])
            visual_feature = visual_feature.repeat_interleave(times, dim=0)
        elif visual_feature.shape[0] > max_K:
            times = math.ceil(visual_feature.shape[0] / max_K)
            visual_feature = visual_feature[::times][:max_K]
        visual_feature = torch.cat([visual_feature, torch.zeros((max_K - visual_feature.shape[0], 45, 1000))], dim=0)

        # get ASR feature
        with open('./data/asr_w2v+mfcc%s.pkl' % vid, 'rb') as f:
            asr_feature = pickle.load(f)  # (num_word, 300+650)
        asr_feature = asr_feature[:max_N]
        actual_N = asr_feature.shape[0]
        asr_feature = torch.cat([asr_feature, torch.zeros((max_N - asr_feature.shape[0], 300+650))], dim=0)

        # get frames mask & ASR words mask
        mask_K = torch.zeros(max_K, dtype=torch.int)
        mask_K[:actual_K] = 1
        mask_N = torch.zeros(max_N, dtype=torch.int)
        mask_N[:actual_N] = 1
        if actual_N == 0:
            mask_N[:] = 1

        return {
            'label': label,
            'caption_feature': caption_feature,
            'visual_feature': visual_feature,
            'asr_feature': asr_feature,
            'mask_K': mask_K,
            'mask_N': mask_N,
        }


class C3DDataset(Dataset):

    def __init__(self, path_vid):
        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)

        self.vid = []
        with open('./data/vids/'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)    
        self.data.reset_index(inplace=True) 

        self.c3dfeapath='./data/c3d/'

    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        label = 1 if item['annotation']=='假' else 0
        label = torch.tensor(label)

        c3d = h5py.File(self.c3dfeapath+vid+".hdf5", "r")[vid]['c3d_features']
        c3d = torch.FloatTensor(c3d)

        return {
            'label': label,
            'c3d': c3d,
        }


class VGGDataset(Dataset):

    def __init__(self, path_vid):
        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)

        self.vid = []
        with open('./data/vids/'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)    
        self.data.reset_index(inplace=True) 
   
        self.framefeapath='./data/ptvgg19_frames/'

        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        label = 1 if item['annotation']=='假' else 0
        label = torch.tensor(label)

        frames=pickle.load(open(os.path.join(self.framefeapath,vid+'.pkl'),'rb'))
        frames=torch.FloatTensor(frames)

        return {
            'label': label,
            'frames': frames,
        }


class BboxDataset(Dataset):
    def __init__(self, path_vid):
        self.data_complete = pd.read_json('./data/data_5500_revised.json',orient='records',dtype=False,lines=True)
        
        self.vid = []
        with open('./data/vids/'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)    
        self.data.reset_index(inplace=True)  

        self.bboxfeapath = './data/bbox_vgg19/'

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        label = 1 if item['annotation']=='假' else 0
        label = torch.tensor(label)

        bbox_vgg = pickle.load(open(os.path.join(self.bboxfeapath,vid+'.pkl'),'rb'))
        bbox_vgg = torch.FloatTensor(bbox_vgg)

        return {
            'label': label,
            'bbox_vgg': bbox_vgg
        }


class Title_W2V_Dataset(Dataset):
    def __init__(self, path_vid, wv_from_text):
        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)

        self.vid = []
        with open('./data/vids/'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)    
        self.data.reset_index(inplace=True)     

        self.wv_from_text = wv_from_text

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        item = self.data.iloc[idx]

        label = 1 if item['annotation']=='假' else 0
        label = torch.tensor(label)

        text = item['description']+' '+item['ocr']
        title_w2v = []
        for word in jieba.cut(text, cut_all=False):
            if self.wv_from_text.__contains__(word):
                try:
                    title_w2v.append(self.wv_from_text[word])
                except:
                    continue

        title_w2v = torch.FloatTensor(title_w2v)
        
        return {
            'label': label,
            'title_w2v': title_w2v,
        }


class CommentsDataset(Dataset):

    def __init__(self, path_vid):
        self.data_complete = pd.read_json('./data/data.json',orient='records',dtype=False,lines=True)

        self.vid = []
        with open('./data/vids/'+path_vid, "r") as fr:
            for line in fr.readlines():
                self.vid.append(line.strip())
        self.data = self.data_complete[self.data_complete.video_id.isin(self.vid)]  
        self.data['video_id'] = self.data['video_id'].astype('category')
        self.data['video_id'].cat.set_categories(self.vid, inplace=True)
        self.data.sort_values('video_id', ascending=True, inplace=True)    
        self.data.reset_index(inplace=True)       

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

        hascomments = self.data['comments'].apply(lambda x:len(x)>0)
        self.data = self.data[hascomments]
        print (self.data.shape)
        
    def __len__(self):
        return self.data.shape[0]
     
    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        vid = item['video_id']

        label = 1 if item['annotation']=='假' else 0
        label = torch.tensor(label)

        comments_inputid = []
        comments_mask = []
        for comment in item['comments']:
            comment_tokens = self.tokenizer(comment, max_length=250, padding='max_length', truncation=True)
            comments_inputid.append(comment_tokens['input_ids'])
            comments_mask.append(comment_tokens['attention_mask'])
        comments_inputid = torch.LongTensor(comments_inputid)
        comments_mask = torch.LongTensor(comments_mask)
        
        comments_like = []
        for num in item['comments_like']:
            num_like = num.split(" ")[0] 
            comments_like.append(str2num(num_like))
        comments_like = torch.tensor(comments_like)

        return {
            'label': label,
            'comments_inputid': comments_inputid,
            'comments_mask': comments_mask,
            'comments_like': comments_like,
        }
