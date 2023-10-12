import numpy as np
from sklearn.cluster import KMeans
import numpy as np
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import k_means
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, v_measure_score
from sklearn.preprocessing import normalize
import time
from sklearn import metrics
import os
# from sklearn.metrics import jaccard_similarity_score
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
files = ['aml', 'breast', 'colon', 'kidney', 'liver', 'lung', 'melanoma', 'ovarian', 'sarcoma','gbm']  #
data_names = ['Transformer']
for f in files:
    for data_name in data_names:
        # encoded_factors=np.loadtxt('./result/cancer_do_cluster/{f}/{d}.txt'.format(f=f, d=data_name))
        encoded_factors=np.loadtxt('./results/cancer/{f}/{d}_10.txt'.format(f=f, d=data_name))
        # savepath = './result/cancer_do_cluster/{f}/{d}_cluster_result.txt'.format(f=f, d=data_name)
        savepath='./results/cancer/{f}/{d}_cluster_result_10.txt'.format(f=f, d=data_name)
        with open(savepath, 'w') as f2:
            print('cancer:{f}\nmethod:{d}'.format(f=f, d=data_name))
            f2.write('cancer:{f}\nmethod:{d}\n'.format(f=f, d=data_name))
            for typenum in range(2,7,1):
                all_silhouette=[]
                all_DBI=[]
                for i in range(1000):
                    clf = KMeans(n_clusters=typenum)#.to(device)
                    clf.fit(encoded_factors)#.to(device)  # 模型训练
                    labels = clf.labels_
                    silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
                    all_silhouette.append(silhouetteScore)
                    davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
                    all_DBI.append(davies_bouldinScore)
                avg_silhouette=np.mean(all_silhouette)
                avg_DBI=np.mean(all_DBI)

                # print("silhouetteScore:", avg_silhouette)
                # print("davies_bouldinScore:", avg_DBI)
                print('k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=typenum, s=avg_silhouette,d=avg_DBI))
                f2.write('zly'*20+'\n')
                f2.write('k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=typenum, s=avg_silhouette,d=avg_DBI))
