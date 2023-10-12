
# DEDUCE: Multi-head attention decoupled contrastive learning to discover cancer subtypes based on multi-omics data

Due to the high heterogeneity and clinical characteristics of cancer, there are significant differences in multi-omics data and clinical features among subtypes of different cancers. Therefore, the identification and discovery of cancer subtypes are crucial for the diagnosis, treatment, and prognosis of cancer. In this study, we proposed a generalization framework based on attention mechanisms for unsupervised contrastive learning to analyze cancer multi-omics data for the identification and characterization of cancer subtypes. The framework contains an unsupervised multi-head attention mechanism that can deeply extract multi-omics data features. Importantly, the proposed framework includes a decoupled contrastive learning model (DEDUCE) based on a multi-head attention mechanism to learn multi-omics data features and clustering and identify new cancer subtypes. This unsupervised contrastive learning method clusters subtypes by calculating the similarity between samples in the feature space and sample space of multi-omics data. Compared to 11 other deep learning models, the DEDUCE model achieved a C-index of 0.002, a Silhouette score of 0.801, and a Davies Bouldin Score of 0.38 on a single-cell multi-omics dataset. On a cancer multi-omics dataset, the DEDUCE model obtained a C-index of 0.016, a Silhouette score of 0.688, and a Davies Bouldin Score of 0.46, and obtained the most reliable cancer subtype clustering results for each type of cancer. Finally, we used the DEDUCE model to reveal six cancer subtypes of AML. By analyzing GO functional enrichment, subtype-specific biological functions and GSEA of AML, we further enhanced the interpretability of cancer subtype analysis based on the generalizable framework of the DEDUCE model. 

## Overview
![输入图片说明](/imgs/2023-10-12/Ea3ZQQ0Ed5ha81Sx.png)


## Introduction

Our full version of the code will be updated after the paper is published!

## Dataset

1.  **Simulated Dataset**: It includes DNA methylation, mRNA gene expression, and protein expression data from 100 samples, with clusters set to 5, 10, or 15.
2.  **Single-cell dataset**: This dataset includes 206 single-cell samples from three cancer cell lines (HTC, Hela, and K562).
3.  **Cancer Multi-Omics Dataset**: This dataset is derived from the cancer multi-omics dataset in The Cancer Genome Atlas (TCGA) and consists of gene expression, DNA methylation, and miRNA expression data. The dataset includes breast cancer (BRCA), glioblastoma (GBM), sarcoma (SARC), lung adenocarcinoma (LUAD), and stomach cancer (STAD) from TCGA. Other cancer types are selected from the baseline dataset, including colon cancer (Colon), acute myeloid leukemia (AML), kidney cancer (Kidney), melanoma, and ovarian cancer.<All datasets can be accessed at http://acgt.cs.tau.ac.il/multi_omic_benchmark/download.html

## Installation

Please install the third-party library first, refer to [libraries.txt](https://github.com/pengsl-lab/DEDUCE/blob/main/libraries.txt).

## Training

The training command is very simple like this:
>python train_cancer_cluster.py

## Inference
>python test_cancer_random.size.py


## Results
Comparison of DEDUCE with 11 other methods on three datasets:
**Simulated Dataset**
![输入图片说明](/imgs/2023-10-12/XoEoGWmW5SGRUB3N.jpeg)
**Single-cell dataset**:
![输入图片说明](/imgs/2023-10-12/fhXkJvJpgVsskAL1.jpeg)
**Cancer Multi-Omics Dataset**: 
![输入图片说明](/imgs/2023-10-12/Cb7WuTBaHMI0ag4R.jpeg)


## Bioinformatics Analysis
For example, Acute Myeloid Leukemia (AML) analysis：
Find differentially expressed genes , gene enrichment analysis and GSEA analysis of different cancer subtypes:
differential gene expression
![differential gene expression](/imgs/2023-10-12/s6jLdf2mXTIJBYJI.png)
GO enrichment analysis
![输入图片说明](/imgs/2023-10-12/F6SmxDWtm6YjISO5.png)
GSEA analysis:
![输入图片说明](/imgs/2023-10-12/MGObUt389JhFs0vW.png)


## Citation

If you find our paper/code are helpful, please consider citing:
```
@article{Multi-Head Attention Mechanism Learning for Cancer New Subtypes and Treatment Based on Cancer Multi-Omics Data
  title={Reduce Information Loss in Transformers for Pluralistic Image Inpainting},
  author={Liangrui Pan, Dazhen Liu, Yutao Dou, Lian Wang, Zhichao Feng, Pengfei Rong, Liwen Xu, Shaoliang Peng},
  booktitle={arxiv},
  year={2023}
}
```

## The data and code will be updated after the paper is officially published!

