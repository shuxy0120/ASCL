# ASCL: Adaptive Self-supervised Counterfactual Learning for Robust Visual Question Answering
## Prerequisites

- python 3.7.15
- pytorch 1.9.1
- pytorch-metric-learning 2.1.0
- tdqm
- Pillow
- transformers

## Download and preprocess the data

```
cd data 
bash download.sh
python create_dictionary.py
python compute_softscore.py cp_v2
cd ..
```

the rest data and trained model can also be obtained from [BaiduYun](https://pan.baidu.com/s/1dR-IDEW3dIggVWzq4b3r0g )(passwd:uglj) 

unzip rcnn_features.zip into data/rcnn/
