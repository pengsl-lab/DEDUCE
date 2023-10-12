import argparse
from dataloader import simulloader, Classloader
import torch
import torch.nn as nn
from models.CNN import NeuralNetwork
from sklearn.preprocessing import normalize
from models.network import Network
from modules import contrastive_loss
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import os
import numpy as np
from models.cancer_classification_attention import ED5
import pandas as pd
from modules.CKloss import LinearCritic, ConditionalSamplingLoss
from tqdm import tqdm
from modules.lars import LARS
from torch.utils.data import DataLoader
import time
from dataloader import MyDataSet
import torch as nn
from torch.utils.data import random_split
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from sklearn.metrics import classification_report
from models.cancer_classification_attention import ED5 #SPTransformer,

def draw_fig(list, cancertype, epoch):
    x1 = range(0, epoch + 1)
    print(x1)
    y1 = list
    if not os.path.exists('./results/cancer_classification/{}'.format(cancertype)):
        os.makedirs('./results/cancer_classification/{}'.format(cancertype))
        # open('./results/{}/{}'.format(datatypes, typenumbers), 'w')
    # np.savetxt('./results/{}/{}/Train_loss.png'.format(datatypes, typenumbers), 'w')
    save_file = './results/cancer_classification/{}/Train_loss.png'.format(cancertype)
    plt.cla()
    plt.title('Train loss vs. epoch', fontsize=20)
    plt.plot(x1, y1, '.-')
    plt.xlabel('epoch', fontsize=20)
    plt.ylabel('Train loss', fontsize=20)
    plt.grid()
    plt.savefig(save_file)
    plt.show()

def save_model(model_path, model, optimizer, current_epoch):
    out = os.path.join(model_path, "checkpoint_{}.tar".format(current_epoch))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': current_epoch}
    torch.save(state, out)

parser = argparse.ArgumentParser(description='PyTorch MOCK.')
#dataloader
parser.add_argument('--datatypes', default="equal", choices={"equal","heterogeneous"}, help='base learning rate, rescaled by batch_size/256')
parser.add_argument("--typenumbers", default=5, choices={5, 10, 15}, help='the cluster number parameter')

#model
parser.add_argument('--model', default='Transformer', choices= {'Transformer, CNN'}, help='train model')
parser.add_argument('--seed', default=21, help='dataset')
parser.add_argument('--workers', default=8, help='InfoNCE temperature')
parser.add_argument("--start_epoch", default=0, help='Training batch size')
parser.add_argument("--epochs", default=400, help='Number of training epochs')
parser.add_argument("--feature_dim", default=5, help='input')



#train
parser.add_argument("--Train", default=True, choices={False, True}, help='yes or not')
parser.add_argument("--batch_size", default=64, help='input')
parser.add_argument("--learning_rate", default=0.0003, help=" the learning rate")
parser.add_argument("--weight_decay", default=0., help="weight_decay")
parser.add_argument("--instance_temperature", default=0.5, help='instance_temperature')
parser.add_argument("--cluster_temperature", default=1.0, help='cluster_temperature')


# args = parser.parse_args()
if __name__ == '__main__':

    # for ki in range(4):
    args = parser.parse_args()
    if args.Train ==True:
        #read data
        files = ['breast', 'gbm', 'colon', 'lung', 'sarcoma']# ,
        for f in files:
            if f == 'breast':
                class_num = 5
                print(class_num)
            elif f == 'gbm':
                class_num = 5
            elif f == 'sarcoma':
                class_num = 6
            elif f == 'lung':
                class_num = 4
            elif f == 'colon':
                class_num = 2
            else:
                print("not have classification number")
            datapath = 'data/cancer_d2d/{f}'.format(f=f)
            omics1 = np.loadtxt('{}/after_log_exp.txt'.format(datapath), str)
            omics1 = np.delete(omics1, 0, axis=1)
            # omics1 = np.transpose(omics1)
            omics1 = omics1.astype(np.float)
            omics1 = normalize(omics1, axis=0, norm='max')
            print(omics1.shape)
            omics2 = np.loadtxt('{}/after_log_mirna.txt'.format(datapath), str)
            omics2 = np.delete(omics2, 0, axis=1)
            # omics2 = np.transpose(omics2)
            omics2 = omics2.astype(np.float)
            omics2 = normalize(omics2, axis=0, norm='max')
            print(omics2.shape)
            omics3 = np.loadtxt('{}/after_methy.txt'.format(datapath), str)
            omics3 = np.delete(omics3, 0, axis=1)
            # omics3 = np.transpose(omics3)
            omics3 = omics3.astype(np.float)
            omics3 = normalize(omics3, axis=0, norm='max')
            print(omics3.shape)
            labels = np.loadtxt('{datapath}/after_labels.txt'.format(datapath=datapath), str)
            labels = np.delete(labels, 0, axis=1)
            labels = labels.astype(np.int)
            omics = np.concatenate((omics1, omics2, omics3), axis=1)
            print("data loading ...")
            train_data_set = MyDataSet(omics, labels)
            # test_data_set = MyDataSet(test_X, test_y)
            train_dataloader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=False)

            # device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # model.to(device)  # 等价于 model = model.to(device)
            # data = data.to(device)  # 注意：数据必须要有赋值，不能写成 data.to(device)

            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            #模型的存储位置
            model_path = './save/cancer/classification_models_{}'.format(f)
            if not os.path.exists(model_path):
                os.makedirs(model_path)

            # 调用模型
            print('==> Building model..')
            ##############################################################
            # Encoder
            ##############################################################
            if args.model == 'CNN':
                model = NeuralNetwork(out_dim = class_num)
            elif args.model == 'Transformer':
                model = ED5(input_dim =omics.shape[1] ,  out_dim = class_num)
            else:
                raise ValueError("Bad architecture specification")

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            if torch.cuda.device_count() > 1:
                print("Let's use ", torch.cuda.device_count(), "GPUs.")
                model = torch.nn.DataParallel(model)
            model.to(device)  # 等价于 model = model.to(device)

            # model = model.to(device)
            ##打印模型
            print("=======", model)

            # critic = LinearCritic(latent_dim=1000, temperature=1).to(device)
            ###设置优化器
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
            # optimizer = LARS(list(model.parameters()) + list(critic.parameters()), lr=args.learning_rate, eta=1e-3,
            #                          momentum=0.9, weight_decay=1e-4, max_epoch=200)
            # loss_device = device

            loss_fn = torch.nn.CrossEntropyLoss()
            # loss_fn = contrastive_loss.DCL(temperature=0.5, weight_fn=None)
            # criterion = ConditionalSamplingLoss(mode='hardnegatives',
            #              temp_z=0.1, scale=1, lambda_=0.1,
            #              weight_clip_threshold=1e-6, distance_mode='polynomial', inverse_device='cpu', inverse_gradient=False)
            ###设置训练的过程 返回损失函数
            ##训练部分
            def train():
                loss_epoch = 0
                t = tqdm(enumerate(train_dataloader), desc='Loss: **** ', total=len(train_dataloader), bar_format='{desc}{bar}{r_bar}')
                for step, (X, y) in t:
                    optimizer.zero_grad()
                    # X = X.float().to(device)
                    # y = y.to(device)
                    X = X.unsqueeze(1).to(device).float()
                    y = y.squeeze(1).long().to(device).float()
                    pred = model(X)#.to(device)
                    loss = loss_fn(pred, y.long())
                    loss.backward()
                    optimizer.step()
                    loss_epoch += loss.item()
                    t.set_description('Loss: %.3f ' % (loss_epoch / (step + 1)))

                return loss_epoch

            # 开始训练模型
            logger = SummaryWriter(log_dir="./log")
            train_loss = []
            eval_loss = []
            min_loss = 1000
            for epoch in range(args.start_epoch, args.epochs + 1):
                lr = optimizer.param_groups[0]["lr"]
                #train
                train_loss_epoch = train()
                train_loss.append(train_loss_epoch)
                logger.add_scalar("train loss", train_loss_epoch)

                if epoch % 20 == 0:
                    print(f"Epoch [{epoch}/{args.epochs}]\t Loss: {train_loss_epoch}")
            # if train_loss[-1] < min_loss:
            #     min_loss = train_loss
            #     print("save model")
            save_model(model_path, model, optimizer, args.epochs)
            draw_fig(train_loss, f, epoch)



    else:
    ############测试模型

        files = ['breast', 'gbm', 'sarcoma', 'lung', 'colon']
        for f in files:
            if f == 'breast':
                class_num = 5
            elif f == 'gbm':
                class_num = 5
            elif f == 'sarcoma':
                class_num = 6
            elif f == 'lung':
                class_num = 4
            elif f == 'colon':
                class_num = 2
            else:
                print("not have classification number")
            savepath = './results/cancer/{}/classification_resutts.txt'.format(f)
            with open(savepath, 'w') as f2:
                datapath = 'data/cancer_d2d/{f}'.format(f=f)
                omics1 = np.loadtxt('{}/after_log_exp.txt'.format(datapath), str)
                omics1 = np.delete(omics1, 0, axis=1)
                # omics1 = np.transpose(omics1)
                omics1 = omics1.astype(np.float)
                omics1 = normalize(omics1, axis=0, norm='max')
                print(omics1.shape)
                omics2 = np.loadtxt('{}/after_log_mirna.txt'.format(datapath), str)
                omics2 = np.delete(omics2, 0, axis=1)
                # omics2 = np.transpose(omics2)
                omics2 = omics2.astype(np.float)
                omics2 = normalize(omics2, axis=0, norm='max')
                print(omics2.shape)
                omics3 = np.loadtxt('{}/after_methy.txt'.format(datapath), str)
                omics3 = np.delete(omics3, 0, axis=1)
                # omics3 = np.transpose(omics3)
                omics3 = omics3.astype(np.float)
                omics3 = normalize(omics3, axis=0, norm='max')
                print(omics3.shape)
                labels = np.loadtxt('{datapath}/after_labels.txt'.format(datapath=datapath), str)
                labels = np.delete(labels, 0, axis=1)
                labels = labels.astype(np.int)
                omics = np.concatenate((omics1, omics2, omics3), axis=1)
                print("data loading ...")
                test_data_set = MyDataSet(omics, labels)
                # test_data_set = MyDataSet(test_X, test_y)
                test_dataloader = DataLoader(test_data_set, batch_size=args.batch_size, shuffle=False)
                # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                if args.model == 'CNN':
                    model = NeuralNetwork()
                elif args.model == 'Transformer':
                    model = ED5(input_dim =omics.shape[1] ,  out_dim = class_num)
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                if torch.cuda.device_count() > 1:
                    print("Let's use ", torch.cuda.device_count(), "GPUs.")
                    model = torch.nn.DataParallel(model)
                model.to(device)  # 等价于 model = model.to(device)
                model_path = './save/cancer/classification_models_{}'.format(f)
                model_fp = os.path.join(model_path, "checkpoint_{}.tar".format(args.epochs))
                model.load_state_dict(torch.load(model_fp, map_location=device.type)['net'], False)
                model.to(device)

                # def test(test_dataloader):
                model.eval()
                y_true = []
                y_pred = []
                for step, (X, y) in enumerate(test_dataloader):
                    X = X.unsqueeze(1).to(device).float()
                    y = y.squeeze(1).long().to(device).float()
                    pred = model(X)  # .to(device)
                    # input = x.float().to(device)
                    # y = y.to(device)
                    y = y.detach()
                    y_true.extend(y.cpu().detach().numpy())
                    out = model(X)
                    pre = torch.argmax(out, dim=1)
                    # _, pre = torch.max(out.data, 1)
                    pre = pre.detach()
                    y_pred.extend(pre.cpu().detach().numpy())
                    print(y_true)
                    print(y_pred)
                    # return y_true, y_pred

                y_true = torch.tensor(y_true, device='cpu')
                y_true.cpu().detach().numpy()
                y_pred = torch.tensor(y_pred, device='cpu')
                y_pred.cpu().detach().numpy()
                acc = accuracy_score(y_true, y_pred)
                f1_macro = f1_score(y_true, y_pred, average='macro')
                # f1_micro=f1_score(y_true, y_pred, average='micro')
                f1_weighted = f1_score(y_true, y_pred, average='weighted')

                all_acc=[]
                all_f1_macro=[]
                all_f1_weighted=[]
                all_acc.append(acc)
                all_f1_macro.append(f1_macro)
                all_f1_weighted.append(f1_weighted)

                print(classification_report(y_true, y_pred))
                # break
                # print_precison_recall_f1(y_true, y_pred)
                print('caicai' * 20)
                print(
                    'acc:{all_acc}\nf1_macro:{all_f1_macro}\nf1_weighted:{all_f1_weighted}\n'. \
                        format(all_acc=all_acc, all_f1_macro=all_f1_macro, all_f1_weighted=all_f1_weighted))
                avg_acc = np.mean(all_acc)
                avg_f1_macro = np.mean(all_f1_macro)
                avg_f1_weighted = np.mean(all_f1_weighted)

                print(
                    'acc:{avg_acc}\nf1_macro:{avg_f1_macro}\nf1_weighted:{avg_f1_weighted}\n'. \
                        format(avg_acc=avg_acc, avg_f1_macro=avg_f1_macro, avg_f1_weighted=avg_f1_weighted))


