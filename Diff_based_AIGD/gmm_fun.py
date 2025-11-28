import torch
from .models import backbone
import time
import numpy as np
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from .utils import *
import pickle, time
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from sklearn.mixture import GaussianMixture

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.cuda.set_device(device)

def load_net(checkpoint_path):
    DiffusionExtractor = backbone()
    checkpoint = torch.load(checkpoint_path)
    DiffusionExtractor.load_state_dict(checkpoint,strict=False)
    DiffusionExtractor.to(device)
    # DiffusionExtractor = DiffusionExtractor.cuda(device)
    DiffusionExtractor.eval()
    if next(DiffusionExtractor.parameters()).is_cuda:
        print("Model is on CUDA.")
    else:
        print("Model is not on CUDA.")
    return DiffusionExtractor


def img2f(net,dataset):
    Fea_all = []
    for data in tqdm(dataset):
        with torch.no_grad():
            inputdata = data.cuda(device)
            f = net(inputdata.unsqueeze(0))
            Fea_all.append(f.cpu())
    Fea_all = torch.cat(Fea_all, dim=0)
    # print('train shape:{}'.format(Fea_all.shape))
    return Fea_all
    

def train_GMM_sklearn(features,savepath):
    gmm = GaussianMixture(n_components=6 ,init_params='k-means++', random_state=42)
    print(features.shape)
    start_time = time.time()
    try:
        gmm.fit(features.detach().cpu().numpy())
    except:
        gmm.fit(features.numpy())
    end_time = time.time()
    runtime = end_time - start_time
    hours = int(runtime // 3600)
    minutes = int((runtime % 3600) // 60)
    seconds = int(runtime % 60)
    print("time: {}hr {}min {}s".format(hours, minutes, seconds))
    with open(savepath, 'wb') as file:
        pickle.dump(gmm, file)




def val_GMM_sklearn(modelpath,features_baseline, rate):
    with open(modelpath, 'rb') as file:
        gmm = pickle.load(file)
    print(f'real: {len(features_baseline)}')
    real_logp = []
    log_likelihoods_real = gmm.score_samples(features_baseline.cpu())
    real_logp.extend(log_likelihoods_real.tolist())
    real_logp = sorted(real_logp)
    threshold_index = int(len(real_logp) * rate) 
    threshold = real_logp[threshold_index]
    print('threshold:{}'.format(threshold))
    return threshold



def f2loglikelihood(modelpath, features):
    with open(modelpath, 'rb') as file:
        gmm = pickle.load(file)
    logp = []
    log_likelihoods = gmm.score_samples(features)
    logp.extend(log_likelihoods.tolist())
    return logp


def compute_mAP_acc_basic(val_name,threshold,loglikelihood_real,loglikelihood_fake):
    logp = []
    label = []
    real_label = [1] * len(loglikelihood_real)
    fake_label = [0] * len(loglikelihood_fake)
    logp.extend(loglikelihood_real)
    logp.extend(loglikelihood_fake)
    label.extend(real_label)
    label.extend(fake_label)
    label, logp = label, logp
    pred = []
    for item in logp:
        if item > threshold:
            pred.append(1)
        else:
            pred.append(0)
    acc = accuracy_score(label, pred)* 100
    map = average_precision_score(label, logp) * 100
    t = np.percentile(loglikelihood_fake, 97)
    b = np.percentile(loglikelihood_real, 3)
    print('val:{} | acc:{:.2f} mAP:{:.2f} | fake-top-value:{:.2f} | real-bottom-value:{:.2f}'.format(val_name,acc,map,t,b))
    return acc, map


def compute_mAP_acc_fusing(val_name,threshold_bc,threshold_base,loglikelihood_real_bc,loglikelihood_fake_bc,loglikelihood_real_base,loglikelihood_fake_base):
    logp = []
    label = []
    real_label = [1] * len(loglikelihood_real_bc)
    fake_label = [0] * len(loglikelihood_fake_bc)
    logp.extend(loglikelihood_real_bc)
    logp.extend(loglikelihood_fake_bc)
    label.extend(real_label)
    label.extend(fake_label)
    label_bc, logp_bc = label, logp
    logp = []
    label = []
    real_label = [1] * len(loglikelihood_real_base)
    fake_label = [0] * len(loglikelihood_fake_base)
    logp.extend(loglikelihood_real_base)
    logp.extend(loglikelihood_fake_base)
    label.extend(real_label)
    label.extend(fake_label)
    label_base, logp_base = label, logp
    pred = []
    for item_bc,item_base in zip(logp_bc,logp_base):
        if item_bc > threshold_bc and item_base > threshold_base:
            pred.append(1)
        else:
            pred.append(0)
    acc = accuracy_score(label, pred)* 100
    map = average_precision_score(label, logp) * 100
    print('val:{} | acc:{:.2f} | map:{:.2f}'.format(val_name,acc,map))
    return acc,map