import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, dropout):
        super(MLP, self).__init__()
        layers = list()
        curr_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(curr_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout))
            curr_dim = hidden_dim
        layers.append(nn.Linear(curr_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, input):
        return self.mlp(input)

class MaskAvg(nn.Module):
    def __init__(self):
        super(MaskAvg, self).__init__()

    def forward(self, input, mask):
        score = torch.ones((input.shape[0], input.shape[1]), device=input.device)
        score = score.masked_fill(mask == 0, float('-inf'))
        score = torch.softmax(score, dim=-1).unsqueeze(1)
        output = torch.matmul(score, input).squeeze(1)
        return output

class CVRL(nn.Module):
    def __init__(self, d_w, d_f, obj_num, gru_dim):
        super(CVRL, self).__init__()
        self.gru = nn.GRU(d_w, gru_dim, batch_first=True, bidirectional=True)

        self.linear_r = nn.Linear(d_f, 1)
        self.linear_h = nn.Linear(2*gru_dim, obj_num)

    def forward(self, caption_feature, visual_feature):
        # IN: caption_feature: (bs, K, S, d_w), visual_feature: (bs, K, obj_num, d_f)
        # OUT: frame_visual_rep: (bs, K, d_f)
        encoded_caption, _ = self.gru(caption_feature.view(-1, caption_feature.shape[-2], caption_feature.shape[-1]))  # (bs*K, S, 2*gru_dim)
        encoded_caption = encoded_caption.view(-1, caption_feature.shape[-3], caption_feature.shape[-2], encoded_caption.shape[-1])  # (bs, K, S, 2*gru_dim)
        frame_caption_rep = encoded_caption.max(dim=2).values  # (bs, K, 2*gru_dim)

        alpha = self.linear_r(visual_feature).squeeze() + self.linear_h(frame_caption_rep)  # (bs, K, obj_num)
        alpha = torch.softmax(torch.tanh(alpha), dim=-1).unsqueeze(dim=-2)  # (bs, K, 1, obj_num)
        frame_visual_rep = alpha.matmul(visual_feature)  # (bs, K, 1, d_f)
        frame_visual_rep = frame_visual_rep.squeeze()  # (bs, K, d_f)
        return frame_visual_rep

class ASRL(nn.Module):
    def __init__(self, d_w, gru_dim):
        super(ASRL, self).__init__()
        self.gru = nn.GRU(d_w, gru_dim, batch_first=True, bidirectional=True)

    def forward(self, asr_feature):
        # IN: asr_feature: (bs, N, d_w)
        # OUT: text_audio_rep: (bs, N, 2*gru_dim)
        text_audio_rep, _ = self.gru(asr_feature)
        return text_audio_rep

class VCIF(nn.Module):
    def __init__(self, d_f, d_w, d_H, gru_f_dim, gru_w_dim, dropout):
        super(VCIF, self).__init__()

        self.param_D = nn.Parameter(torch.empty((d_f, d_w)))
        self.param_Df = nn.Parameter(torch.empty((d_f, d_H)))
        self.param_Dw = nn.Parameter(torch.empty((d_w, d_H)))
        self.param_df = nn.Parameter(torch.empty(d_H))
        self.param_dw = nn.Parameter(torch.empty(d_H))

        self.gru_f = nn.GRU(d_f, gru_f_dim, batch_first=True)
        self.gru_w = nn.GRU(d_w, gru_w_dim, batch_first=True)
        self.mask_avg = MaskAvg()
        self.dropout = nn.Dropout(p=dropout)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.param_D)
        nn.init.xavier_uniform_(self.param_Df)
        nn.init.xavier_uniform_(self.param_Dw)
        nn.init.uniform_(self.param_df)
        nn.init.uniform_(self.param_dw)

    def forward(self, frame_visual_rep, text_audio_rep, mask_K, mask_N):
        # IN: frame_visual_rep: (bs, K, d_f), text_audio_rep: (bs, N, d_w)
        # OUT: video_rep: (bs, gru_f_dim + gru_w_dim)
        affinity_matrix = torch.tanh(frame_visual_rep.matmul(self.param_D).matmul(text_audio_rep.transpose(-1, -2)))
        affinity_matrix = self.dropout(affinity_matrix)

        frame_co_att_map = torch.tanh(frame_visual_rep.matmul(self.param_Df) + affinity_matrix.matmul(text_audio_rep).matmul(self.param_Dw))
        word_co_att_map = torch.tanh(text_audio_rep.matmul(self.param_Dw) + affinity_matrix.transpose(-1, -2).matmul(frame_visual_rep).matmul(self.param_Df))
        frame_co_att_map = self.dropout(frame_co_att_map)
        word_co_att_map = self.dropout(word_co_att_map)

        frame_att_weight = torch.softmax(frame_co_att_map.matmul(self.param_df), dim=-1)
        word_att_weight = torch.softmax(word_co_att_map.matmul(self.param_dw), dim=-1)

        frame_visual_weighted_rep = frame_att_weight.unsqueeze(dim=-1) * frame_visual_rep
        text_audio_weighted_rep = word_att_weight.unsqueeze(dim=-1) * text_audio_rep

        encoded_visual_rep, _ = self.gru_f(frame_visual_weighted_rep)
        encoded_speech_rep, _ = self.gru_w(text_audio_weighted_rep)

        visual_rep = self.mask_avg(encoded_visual_rep, mask_K)  # (bs, gru_f_dim)
        speech_rep = self.mask_avg(encoded_speech_rep, mask_N)  # (bs, gru_w_dim)

        video_rep = torch.cat([visual_rep, speech_rep], dim=-1)
        return video_rep

class TikTecModel(nn.Module):
    def __init__(self, word_dim=300, mfcc_dim=650, visual_dim=1000, obj_num=45, CVRL_gru_dim=200, ASRL_gru_dim=500, VCIF_d_H=200, VCIF_gru_f_dim=200, VCIF_gru_w_dim=100, VCIF_dropout=0.2, MLP_hidden_dims=[512], MLP_dropout=0.2):
        super(TikTecModel, self).__init__()
        self.CVRL = CVRL(d_w=word_dim, d_f=visual_dim, obj_num=obj_num, gru_dim=CVRL_gru_dim)
        self.ASRL = ASRL(d_w=(word_dim + mfcc_dim), gru_dim=ASRL_gru_dim)
        self.VCIF = VCIF(d_f=visual_dim, d_w=2*ASRL_gru_dim, d_H=VCIF_d_H, gru_f_dim=VCIF_gru_f_dim, gru_w_dim=VCIF_gru_w_dim, dropout=VCIF_dropout)
        self.MLP = MLP(VCIF_gru_f_dim + VCIF_gru_w_dim, MLP_hidden_dims, 2, MLP_dropout)

    def forward(self, **kwargs):
        # IN:
        #   caption_feature: (bs, K, S, word_dim) = (bs, 200, 100, 300)
        #   visual_feature: (bs, K, obj_num, visual_dim) = (bs, 200, 45, 1000)
        #   asr_feature: (bs, N, word_dim + mfcc_dim) = (bs, 500, 300 + 650)
        #   mask_K: (bs, K) = (bs, 200)
        #   mask_N: (bs, N) = (bs, 500)
        # OUT: (bs, 2)
        caption_feature = kwargs['caption_feature']
        visual_feature = kwargs['visual_feature']
        asr_feature = kwargs['asr_feature']
        mask_K = kwargs['mask_K']
        mask_N = kwargs['mask_N']

        frame_visual_rep = self.CVRL(caption_feature, visual_feature)
        text_audio_rep = self.ASRL(asr_feature)
        video_rep = self.VCIF(frame_visual_rep, text_audio_rep, mask_K, mask_N)
        output = self.MLP(video_rep)
        return output
