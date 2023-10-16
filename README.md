# FakeSV
Official repository for ["***FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms***"](https://arxiv.org/abs/2211.10973), AAAI 2023. (Please note that the arxiv version is more complete.)
- **Dataset**: The video ID (which can be used to infer the video URL) and corresponding annotations have been released. Also, we provide two data split used in the paper, i.e. event-based and temporal. 
- **Models**: We reproduce some SOTA methods on fake news video detection to provide benchmark results for FakeSV. Codes for our proposed model SV-FEND and other methods are provided. 

### Environment
Anaconda 4.13.0, python 3.8.5, pytorch 1.10.1 and cuda 11.7. For other libs, please refer to the file requirements.txt.

### Application for Data Use
Please sign [this agreement](https://drive.google.com/file/d/1Ozj5OOYoDYnDznDLAECBvlGIDtyGl6Vz) and send the signed copy through your **institutional email** to pengqi.qp@gmail.com.

### Data Processing
[video-subtitle-extractor](https://github.com/YaoFANGUK/video-subtitle-extractor)

[bert-base-chinese](https://github.com/google-research/bert)

[VGG19](https://pytorch.org/vision/main/models)

[C3D](https://github.com/yyuanad/Pytorch_C3D_Feature_Extractor)

[VGGish](https://github.com/harritaylor/torchvggish)

[MoviepPy](https://github.com/Zulko/moviepy)

You could use the above repositories to extract features by yourself, or use our pre-extracted features ([VGG19](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/ptvgg19_frames.zip)/[C3D](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/c3d.zip)/[VGGish](https://huggingface.co/datasets/MischaQI/FakeSV/blob/main/dict_vid_audioconvfea.pkl)). Besides, we also provide [five-fold checkpoints](https://huggingface.co/datasets/MischaQI/FakeSV/tree/main/checkpoints) for comparison. 

### Citation
```
@inproceedings{fakesv, 
title={FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection  on Short Video Platforms}, 
author={Qi, Peng and Bu, Yuyan and Cao, Juan and Ji, Wei and Shui, Ruihao and Xiao,  Junbin and Wang, Danding and Chua, Tat-Seng}, 
booktitle={Proceedings of the AAAI Conference on Artificial Intelligence}, 
year={2023}, 
organization={AAAI} 
} 
```

[Related Survey (will be presented at ACM MM 2023)](https://arxiv.org/abs/2302.03242) and [Companion GitHub Repository](https://github.com/ICTMCG/Awesome-Misinfo-Video-Detection) :
```
@inproceedings{mvdsurvey, 
title={Combating Online Misinformation Videos: Characterization, Detection, and Future Directions}, 
author={Bu, Yuyan and Sheng, Qiang and Cao, Juan and Qi, Peng and Wang, Danding and Li, Jintao}, 
booktitle={Proceedings of the 31st ACM International Conference on Multimedia}, 
year={2023},
doi={10.1145/3581783.3612426},
publisher = {Association for Computing Machinery},
} 
```
