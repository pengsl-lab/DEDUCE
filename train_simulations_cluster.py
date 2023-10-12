import argparse
from dataloader import simulloader
import torch
from models.AEclass import AE
# from models.ae import AE
from models.DAEclass import DAE, ED2
from models.VAEclass import VAE, ED1
from models.SVAEclass import SVAE, ED3
from models.network import Network
from modules import contrastive_loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from modules.CKloss import LinearCritic, ConditionalSamplingLoss
from tqdm import tqdm
from modules.lars import LARS
from modules.MMCL import CKloss, NT_Xent
from models.attention import Transformer, ED4
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, v_measure_score
from sklearn.cluster import KMeans
import time
from sklearn import metrics
import os


def draw_fig(list, datatypes, typenumbers, epoch):
    x1 = range(0, epoch + 1)
    print(x1)
    y1 = list
    if not os.path.exists('./results/simulations/{}/{}'.format(datatypes, typenumbers)):
        os.makedirs('./results/simulations/{}/{}'.format(datatypes, typenumbers))
        # open('./results/{}/{}'.format(datatypes, typenumbers), 'w')
    # np.savetxt('./results/{}/{}/Train_loss.png'.format(datatypes, typenumbers), 'w')
    save_file = './results/simulations/{}/{}/Train_loss.png'.format(datatypes, typenumbers)
    plt.cla()
    plt.title('Train loss vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig(save_file)
    plt.show()


def inference(loader, model, device):
    model.eval()
    cluster_vector = []
    feature_vector = []
    for step, x in enumerate(loader):
        x = x.float().to(device)
        with torch.no_grad():
            c, h = model.forward_cluster(x)
        c = c.detach()
        h = h.detach()
        cluster_vector.extend(c.cpu().detach().numpy())
        feature_vector.extend(h.cpu().detach().numpy())
    cluster_vector = np.array(cluster_vector)
    feature_vector = np.array(feature_vector)
    print("Features shape {}".format(feature_vector.shape))
    return cluster_vector, feature_vector


def save_model(model_path, model, optimizer, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)


parser = argparse.ArgumentParser(description='PyTorch MOCK.')
# dataloader
parser.add_argument('--datatypes', default="heterogeneous", choices={"equal", "heterogeneous"},
                    help='base learning rate, rescaled by batch_size/256')
parser.add_argument("--typenumbers", default=5, choices={5, 10, 15}, help='the cluster number parameter')

# model
parser.add_argument('--model', default='Transformer', choices={'Transformer'}, help='train model')
parser.add_argument('--seed', default=21, help='dataset')
parser.add_argument('--workers', default=8, help='InfoNCE temperature')
parser.add_argument("--start_epoch", default=0, help='Training batch size')
parser.add_argument("--epochs", default=1000, help='Number of training epochs')
parser.add_argument("--feature_dim", default=5, choices={5, 10, 15}, help='input')

# train
parser.add_argument("--Train", default=True, choices={False, True}, help='yes or not')
parser.add_argument("--batch_size", default=100, help='input')
parser.add_argument("--learning_rate", default=0.0001, help=" the learning rate")
parser.add_argument("--weight_decay", default=0., help="weight_decay")
parser.add_argument("--instance_temperature", default=0.5, help='instance_temperature')
parser.add_argument("--cluster_temperature", default=1.0, help='cluster_temperature')

# args = parser.parse_args()
if __name__ == '__main__':
    args = parser.parse_args()
    # read data
    if args.Train == True:
        traindata = simulloader(args.datatypes, args.typenumbers, True)

        # device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 模型的存储位置
        model_path = './save/{}_{}'.format(args.datatypes, args.typenumbers)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        # 调用模型
        print('==> Building model..')
        ##############################################################
        # Encoder
        ##############################################################
        if args.model == 'AE':
            model = Network(AE(), args.feature_dim, args.typenumbers)
        elif args.model == 'VAE':
            model = VAE(ED1(), args.feature_dim, args.typenumbers)
        elif args.model == 'DAE':
            model = DAE(ED2(), args.feature_dim, args.typenumbers)
        elif args.model == 'SVAE':
            model = SVAE(ED3(), args.feature_dim, args.typenumbers)
        elif args.model == 'Transformer':
            model = Transformer(ED4(out_dim=20), args.feature_dim, args.typenumbers)
        # elif args.arch == 'LeNet':
        #     net = LeNet()
        else:
            raise ValueError("Bad architecture specification")

        model = model.to(device)
        ##打印模型
        print("=======", model)

        # critic = LinearCritic(latent_dim=1000, temperature=1).to(device)
        ##设置优化器
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, eps= 1e-3)#,weight_decay=args.weight_decay)
        # optimizer = LARS(list(model.parameters()) + list(critic.parameters()), lr=args.learning_rate, eta=1e-3,
        #                  momentum=0.9, weight_decay=1e-4, max_epoch=10000)
        loss_device = device

        # criterion = ConditionalSamplingLoss(mode='hardnegatives',
        #                                     temp_z=0.1, scale=1, lambda_=0.1,
        #                                     weight_clip_threshold=1e-6, distance_mode='RBF', inverse_device='cpu',
        #                                     inverse_gradient=False)


        ###设置训练的过程 返回损失函数
        def train():
            loss_epoch = 0
            t = tqdm(enumerate(traindata), desc='Loss: **** ', total=len(traindata), bar_format='{desc}{bar}{r_bar}')
            for step, x in t:
                optimizer.zero_grad()
                # out = model(x)
                # # critic.train()
                x_i = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
                x_j = (x + torch.normal(0, 1, size=(x.shape[0], x.shape[1]))).float().to(device)
                z_i, z_j, c_i, c_j = model(x_i, x_j)
                batch = x_i.shape[0]
                criterion_instance = contrastive_loss.DCL(temperature=0.5, weight_fn=None)

                # criterion_instance = NT_Xent(args.batch_size, temperature=0.5)
                # raw_scores, pseudotargets = critic(z_i, z_j)
                # loss_instance = criterion(raw_scores, condition1=z_i, condition2=z_i, high_threshold=1, low_threshold=0.2)
                # criterion_instance = CKloss(anchor_count = 2, batch_size=100, reg = 0.1,  kernel = 'poly') #tanh, min, rbf, poly, linear
                criterion_cluster = contrastive_loss.ClusterLoss(args.typenumbers, args.cluster_temperature, loss_device).to(loss_device)
                loss_instance = criterion_instance(z_i, z_j) + criterion_instance(z_j, z_i)

                # loss_instance = criterion_instance(z_i) + criterion_instance(z_j)
                loss_cluster = criterion_cluster(c_i, c_j)
                loss = loss_instance + loss_cluster
                # loss.backward(torch.ones_like(loss))
                loss.backward()
                optimizer.step()
                loss_epoch += loss.item()
                t.set_description('Loss: %.3f ' % (loss_epoch / (step + 1)))

            return loss_epoch


        # 开始训练模型
        logger = SummaryWriter(log_dir="./log")
        loss = []
        for epoch in range(args.start_epoch, args.epochs + 1):
            lr = optimizer.param_groups[0]["lr"]
            loss_epoch = train()
            loss.append(loss_epoch)
            logger.add_scalar("train loss", loss_epoch)
            if epoch % 20 == 0:
                print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch}")
        save_model(model_path, model, optimizer, args.epochs)
        draw_fig(loss, args.datatypes, args.typenumbers, epoch)
    else:
        # 测试模型
        # inference
        dataloader = simulloader(args.datatypes, args.typenumbers, False)

        # load model
        if args.model == 'AE':
            model = Network(AE(), args.feature_dim, args.typenumbers)
        elif args.model == 'VAE':
            model = VAE(ED1(), args.feature_dim, args.typenumbers)
        elif args.model == 'DAE':
            model = DAE(ED2(), args.feature_dim, args.typenumbers)
        elif args.model == 'SVAE':
            model = SVAE(ED3(), args.feature_dim, args.typenumbers)
        elif args.model == 'Transformer':
            model = Transformer(ED4(out_dim=20), args.feature_dim, args.typenumbers)

        model_path = './save/{}_{}'.format(args.datatypes, args.typenumbers)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.epochs))
        model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'])
        model.to(device)

        print("### Creating features from model ###")
        X, h = inference(dataloader, model, device)
        # output = pd.DataFrame(columns=['sample_name', 'dcc'])  # 建立新的DataFrame
        # fea_tmp_file = './data/simulations/{}/{}/clusters.txt'.format(args.datatypes, args.typenumbers)  # 找到癌症聚类的标签文件
        # sample_name = list(pd.read_table(fea_tmp_file).columns)[:1]          ##在文件中找到癌症聚类的标签
        # sample_name = pd.read_table(fea_tmp_file, sep='\t', header=None)
        # sample_name = list(sample_name[1])
        #output['sample_name'] = sample_name  # 文件的第一列名称
        # output['dcc'] = X #+ 1  ##文件的第二列聚类标签
        # out_file = './results/{}/{}_label.txt'.format(args.datatypes, args.model)  # 保存文件的标签
        # output.to_csv(out_file, index=False, sep='\t')
        np.savetxt('./results/simulations/{}/{}/{}_label_{}.txt'.format(args.datatypes, args.typenumbers, args.model, args.typenumbers), X, fmt = '%d')

        np.savetxt('./results/simulations/{}/{}/{}_{}.txt'.format(args.datatypes,args.typenumbers, args.model, args.typenumbers), h)
        # fea_out_file = './results/{}/{}.txt'.format(args.datatypes, args.model)
        # fea = pd.DataFrame(data=h, columns=map(lambda x: 'v' + str(x), range(h.shape[1])))
        # fea.to_csv(fea_out_file, header=True, index=True, sep='\t')

        ###测试

        encoded_factors = h
        resultpath = 'O:\\pytorch projects\\MOCK\\results\\simulations\\{}\\{}'.format(args.datatypes, args.typenumbers)
        datapath = 'data/simulations/{}/{}'.format(args.datatypes, args.typenumbers)
        groundtruth = np.loadtxt('{}/c.txt'.format(datapath))
        groundtruth = list(np.int_(groundtruth))
        if not os.path.exists("{}_Kmeans.txt".format(args.model)):
            open("{}_Kmeans.txt".format(args.model), 'w')
        fo = open("{}_Kmeans.txt".format(args.model), "a")
        # clf = KMeans(n_clusters=args.typenumbers)
        # t0 = time.time()
        # clf.fit(encoded_factors)  # 模型训练
        # km_batch = time.time() - t0  # 使用kmeans训练数据消耗的时间
        #
        # print(args.datatypes, args.typenumbers)
        # print("K-Means算法模型训练消耗时间:%.4fs" % km_batch)

        # 效果评估
        score_funcs = [
            metrics.adjusted_rand_score,  # ARI（调整兰德指数）
            metrics.v_measure_score,  # 均一性与完整性的加权平均
            metrics.adjusted_mutual_info_score,  # AMI（调整互信息）
            metrics.mutual_info_score,  # 互信息
        ]
        # centers = clf.cluster_centers_
        # # print("centers:")
        # # print(centers)
        # print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")
        # labels = clf.labels_
        # print("labels:")
        # print(labels)
        labels = X
        labels = list(np.int_(labels))
        # if not os.path.exists("{}/{}_CL.txt".format(resultpath, args.model)):
        #     open("{}/{}_CL.txt".format(resultpath, args.model), 'w')
        # np.savetxt("{}/{}_CL.txt".format(resultpath, args.model), labels, fmt='%d')
        # print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")
        # 2. 迭代对每个评估函数进行评估操作
        for score_func in score_funcs:
            t0 = time.time()
            km_scores = score_func(groundtruth, labels)
            print(
                "K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (score_func.__name__, km_scores, time.time() - t0))
        t0 = time.time()
        # from jaccard_coefficient import jaccard_coefficient

        # from sklearn.metrics import jaccard_similarity_score
        #
        # jaccard_score = jaccard_similarity_score(groundtruth, labels)  # jaccard_coefficient
        # print("K-Means算法:%s评估函数计算结果值:%.5f；计算消耗时间:%0.3fs" % (
        #     jaccard_similarity_score.__name__, jaccard_score, time.time() - t0))
        silhouetteScore = silhouette_score(encoded_factors, labels, metric='euclidean')
        davies_bouldinScore = davies_bouldin_score(encoded_factors, labels)
        print("silhouetteScore:", silhouetteScore)
        print("davies_bouldinScore:", davies_bouldinScore)
        print("zlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzlyzly")