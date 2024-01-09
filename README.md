# ASCL: Adaptive Self-supervised Counterfactual Learning for Robust Visual Question Answering
## Prerequisites

- python 3.7
- pytorch 1.9.1
- pytorch-metric-learning 2.1.0
- tdqm
- Pillow
- transformers

## Download and preprocess the data

```
cd data 
python compute_softscore.py v2
python compute_softscore.py cp_v1
python compute_softscore.py cp_v2
cd ..
```

the data can also be obtained from [BaiduYun](https://pan.baidu.com/s/11ggV8LD7lmtCsFITSYNMCg)(passwd:et10) 

```
unzip rcnn_features.zip into rcnn/
```

the trained model can be obtained from [BaiduYun](https://pan.baidu.com/s/1MDQwhW40JyGScTBWboD91w)(passwd:6eh9) 
