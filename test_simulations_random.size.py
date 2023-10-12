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


datatypes = ["equal", "heterogeneous"]
typenums = [5, 10, 15]
data_names = ['Transformer'] #'VAE_FCTAE_EM', 'AE_FAETC_EM', 'AE_FCTAE_EM', 'DAE_FAETC_EM', 'DAE_FCTAE_EM', 'SVAE_FCTAE_EM', 'MMDVAE_EM']
for datatype in datatypes:
    for typenum in typenums:
        for data_name in data_names:
            encoded_factors = np.loadtxt(
                './results/simulations/{datatype}/{typenum}/{d}_{typenum}.txt'.format(datatype=datatype, typenum=typenum, d=data_name))
            # np.nan_to_num(encoded_factors)
            datapath='data/simulations/{}/{}'.format(datatype, typenum)
            resultpath = 'O:\\pytorch projects\\MOCK\\results\\simulations\\{}\\{}'.format(datatype, typenum)
            groundtruth = np.loadtxt('{}/c.txt'.format(datapath))
            groundtruth = list(np.int_(groundtruth))
            savepath = "O:\\pytorch projects\\MOCK\\results\\simulations\\{datatype}\\{typenum}\\{d}_cluster_result.txt".format(datatype=datatype, typenum=typenum, d=data_name)
            # if not os.path.exists(savepath):
            #     open(r'savepath', 'w+')
            with open(savepath, "w") as f2:
                print('method:{d}\n'.format(d=data_name))
                f2.write('method:{d}\n'.format(d=data_name))
                for cluster_num in range(2, 16, 1):
                    all_Jaccard = []
                    all_C_index = []
                    all_silhouette = []
                    all_DBI = []
                    for i in range(100):
                        clf = KMeans(n_clusters=cluster_num)
                        clf.fit(encoded_factors)  # 模型训练
                        labels = clf.labels_
                        silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
                        all_silhouette.append(silhouetteScore)
                        davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
                        all_DBI.append(davies_bouldinScore)
                    # avg_Jaccard = np.mean(all_Jaccard)
                    # avg_C_index = np.mean(all_C_index)
                    avg_silhouette = np.mean(all_silhouette)
                    avg_DBI = np.mean(all_DBI)

                    # print("silhouetteScore:", avg_silhouette)
                    # print("davies_bouldinScore:", avg_DBI)
                    print(
                        'k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format( k=cluster_num, s=avg_silhouette,
                                                                                       d=avg_DBI))
                    f2.write('*' * 20 + '\n')
                    f2.write(
                        'k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=cluster_num, s=avg_silhouette,
                                                                                       d=avg_DBI))

            # if not os.path.exists("{}_Kmeans.txt".format(data_name)):
            #     open("{}_Kmeans.txt".format(data_name), 'w')
            # fo = open("{}_Kmeans.txt".format(data_name), "a")
            # clf = KMeans(n_clusters=typenum)
            # t0 = time.time()
            # clf.fit(encoded_factors)  # 模型训练
            # km_batch = time.time() - t0  # 使用kmeans训练数据消耗的时间
            #
            # print(datatype, typenum)
            # print("K-Means算法模型训练消耗时间:%.4fs" % km_batch)
            #
            # # 效果评估
            # score_funcs = [
            #     metrics.adjusted_rand_score,  # ARI（调整兰德指数）
            #     metrics.v_measure_score,  # 均一性与完整性的加权平均
            #     metrics.adjusted_mutual_info_score,  # AMI（调整互信息）
            #     metrics.mutual_info_score,  # 互信息
            # ]
            # centers = clf.cluster_centers_
            # # print("centers:")
            # # print(centers)
            # print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")
            # labels = clf.labels_
            # print("labels:")
            # print(labels)
            # labels = list(np.int_(labels))
            # if not os.path.exists("{}/{}_CL.txt".format(resultpath,data_name)):
            #     open("{}/{}_CL.txt".format(resultpath, data_name), 'w')
            # np.savetxt("{}/{}_CL.txt".format(resultpath, data_name), labels, fmt='%d')
            # print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")
            # # 2. 迭代对每个评估函数进行评估操作
            # for score_func in score_funcs:
            #     t0 = time.time()
            #     km_scores = score_func(groundtruth, labels)
            #     print(
            #         "K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__, km_scores, time.time() - t0))
            # t0 = time.time()
            #
            # # from jaccard_coefficient import jaccard_coefficient
            #
            # # from sklearn.metrics import jaccard_similarity_score
            # #
            # # jaccard_score = jaccard_similarity_score(groundtruth, labels)  # jaccard_coefficient
            # # print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (
            # #     jaccard_similarity_score.__name__, jaccard_score, time.time() - t0))
            # silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
            # davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
            # print("silhouetteScore:", silhouetteScore)
            # print("davies_bouldinScore:", davies_bouldinScore)
            # print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")


# 直接拼接
# files = ['aml', 'breast', 'colon', 'kidney', 'liver', 'lung', 'melanoma', 'ovarian', 'sarcoma','gbm']
# for f in files:
#     datapath='./data/cancer_do_cluster/{f}'.format(f=f)
#     omics1 = np.loadtxt('{}/log_exp_omics.txt'.format(datapath))
#     omics1 = np.transpose(omics1)
#     omics1 = normalize(omics1, axis=0, norm='max')
#     print(omics1.shape)
#     omics2 = np.loadtxt('{}/log_mirna_omics.txt'.format(datapath))
#     omics2 = np.transpose(omics2)
#     omics2 = normalize(omics2, axis=0, norm='max')
#     print(omics2.shape)
#     omics3 = np.loadtxt('{}/methy_omics.txt'.format(datapath))
#     omics3 = np.transpose(omics3)
#     omics3 = normalize(omics3, axis=0, norm='max')
#     print(omics3.shape)
#     omics = np.concatenate((omics1, omics2, omics3), axis=1)
#     encoded_factors=omics
#     savepath='./result/cancer_do_cluster/{f}/Contact_cluster_result.txt'.format(f=f)
#     with open(savepath, 'w') as f2:
#         print('cancer:{f}\nmethod:直接拼接'.format(f=f))
#         f2.write('cancer:{f}\nmethod:直接拼接\n'.format(f=f))
#         for typenum in range(2,7,1):
#             all_silhouette=[]
#             all_DBI=[]
#             for i in range(100):
#                 clf = KMeans(n_clusters=typenum)
#                 clf.fit(encoded_factors)  # 模型训练
#                 labels = clf.labels_
#                 silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
#                 all_silhouette.append(silhouetteScore)
#                 davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
#                 all_DBI.append(davies_bouldinScore)
#             avg_silhouette=np.mean(all_silhouette)
#             avg_DBI=np.mean(all_DBI)

#             # print("silhouetteScore:", avg_silhouette)
#             # print("davies_bouldinScore:", avg_DBI)
#             print('k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=typenum, s=avg_silhouette,d=avg_DBI))
#             f2.write('zly'*20+'\n')
#             f2.write('k:{k}\nsilhouetteScore:{s}\ndavies_bouldinScore:{d}\n'.format(k=typenum, s=avg_silhouette,d=avg_DBI))
