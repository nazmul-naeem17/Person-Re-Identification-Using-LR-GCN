# Person Re-Identification Using Local Relation-Aware GCN

This repository implements the **Local Relation-Aware Graph Convolutional Network (LRGCN)** for person re-identification on the Market-1501 datasets, as described in:

> Yu Lian, Wenmin Huang, Shuang Liu, Peng Guo, Zhong Zhang & Tariq S. Durrani.  
> *Person Re-Identification Using Local Relation-Aware Graph Convolutional Network*,  
> **Sensors** 2023, 23, 8138. : https://doi.org/10.3390/s23198138

A clean, easy-to-understand PyTorch Lightning implementation of the architecture, training, and evaluation.

> Quick start in Google Colab:  
> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]

---

## 📊 Results

| Model                  | Dataset | mAP   | CMC Top-1 | CMC Top-5 |
| ---------------------- | ------- | ----- | --------- | --------- |
| LRGCN (Ours)           | Market  | TBD   | TBD       | TBD       |
| LRGCN (Ours)           | Duke    | TBD   | TBD       | TBD       |


---

## 📐 Model Architecture

<p align="center">
  <i>Local Relation-Aware GCN: pose-aligned pooling + overlap & similarity graphs + SGConv stack</i>
</p>

1. **Backbone:** pre-trained ResNet-50 (up to layer4 without stride downsampling).  
2. **Part pooling:** M = 9 horizontal parts, aligned by a frozen Keypoint R-CNN.  
3. **Overlap graph:** k-NN on flattened part features (k=10).  
4. **Similarity graph:** learned φ/ψ projections + masking (S<0.01→0).  
5. **SGConv stack:** five layers of graph convolutions (512→512→256→256→256).  
6. **Classification:** per-part 256-dim head with cross-entropy + label smoothing.  
7. **ST distribution:** optional spatial-temporal smoothing & reranking.

---

## 🚀 Getting Started

### 1. Clone & Install

```
git clone https://github.com/nazmul-naeem17/Person-Re-Identification-Using-LR-GCN
cd Person-Re-Identification-Using-LR-GCN
pip install -r requirements.txt
```
### 2. Prepare Data
```
data/
  raw/
    Market-1501/
      bounding_box_train/
      bounding_box_test/
      query/
```
### 3. Run Training
### 4. Code Structure
```
.
├── README.md
├── requirements.txt
├── setup.py
├── train.py           
├── data.py           
├── model.py           
├── engine.py           
├── utils.py           
├── metrics.py         
├── re_ranking.py      
└── notebooks/
    └── demo.ipynb
```
## 💡Tips & Troubleshooting
**NaN losses:** ensure you’re using precision=32 on CPU or mixed-16 only on GPU, and that your overlap-graph normalization uses the zero-degree guard.

**Slow training:** disable random erasing, use smaller batch sizes, or precompute part features offline for caching.

**Corrupted images:** skipped automatically by the Dataset loader.

