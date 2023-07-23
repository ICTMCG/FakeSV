# FakeSV
Official repository for ["***FakeSV: A Multimodal Benchmark with Rich Social Context for Fake News Detection on Short Video Platforms***"](https://arxiv.org/abs/2211.10973), AAAI 2023. (Please note that the arxiv version is more complete.)
- **Dataset**: The video ID (which can be used to infer the video URL) and corresponding annotations have been released. Also, we provide two data split used in the paper, i.e. event-based and temporal. 
- **Models**: We reproduce some SOTA methods on fake news video detection to provide benchmark results for FakeSV. Codes for our proposed model SV-FEND and other methods are provided. 

### Environment
Anaconda 4.13.0, python 3.8.5, pytorch 1.10.1 and cuda 11.7. For other libs, please refer to the file requirements.txt.

### Application for Data Use
Please sign [this agreement](https://drive.google.com/file/d/1Ozj5OOYoDYnDznDLAECBvlGIDtyGl6Vz) and send the signed copy to pengqi.qp@gmail.com.

### Data Processing
[video-subtitle-extractor](https://github.com/YaoFANGUK/video-subtitle-extractor)

[bert-base-chinese](https://github.com/google-research/bert)

[VGG19](https://pytorch.org/vision/main/models)

[C3D](https://github.com/yyuanad/Pytorch_C3D_Feature_Extractor)

[VGGish](https://github.com/harritaylor/torchvggish)

[MoviepPy](https://github.com/Zulko/moviepy)


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

[Related Survey](https://arxiv.org/abs/2302.03242) and [Companion GitHub Repository](https://github.com/ICTMCG/Awesome-Misinfo-Video-Detection) :
```
@article{fakesvsurvey,
  title={Online Misinformation Video Detection: A Survey},
  author={Yuyan Bu, Qiang Sheng, Juan Cao, Peng Qi, Danding Wang and Jintao Li},
  journal={arXiv preprint arXiv:2302.03242},
  year={2023}
}
```
