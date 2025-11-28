import torch
import torchvision
from PIL import Image
import os
from torchvision import transforms
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
import json
import random
from scipy.ndimage.filters import gaussian_filter
import cv2
from io import BytesIO
from random import choice
from itertools import islice
import warnings
warnings.filterwarnings("ignore")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from torchvision.transforms import Resize
from transformers import CLIPProcessor, CLIPModel




def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img

def pil_jpg_eval(img, compress_val):
    out = BytesIO()
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img = np.array(img)
    img = Image.fromarray(img)
    out.close()
    return img



jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

def custom_augment(img):

    if random.random() < 0.1:
        size = random.randint(64,128)
        img = torchvision.transforms.Resize((size,size))(img)
    
    img = np.array(img)
    if random.random() < 0.1:
        sig = sample_continuous([0.0,1.0])
        gaussian_blur(img, sig)

    if  random.random() < 0.2:
        method = sample_discrete(['cv2','pil'])
        qual = sample_discrete([i for i in range(90,100)])
        img = jpeg_from_key(img, qual, method)


    return Image.fromarray(img)


class DatasetforDiffusionGranularityBinary():
    def __init__(self, root=None, filter_word = '0_real', filter_mode=True, len_limitation = 2000, eval_noise = None) -> None:
        self.root = root
        self.totensor = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
        
        self.len_limitation = len_limitation
        self.patchsize = 16
        self.randomcrop = torchvision.transforms.RandomCrop(self.patchsize)
        self.filter_word = filter_word
        self.filter_mode = filter_mode
        self.paths = self._preprocess() 
        self.eval_noise = eval_noise

    def _preprocess(self):
        paths = []
        for root,dirs,files in os.walk(self.root):
            if len(files) != 0:
                for file in files:
                    if self.filter_mode == True:
                        if self.filter_word in root:
                            paths.append(root+'/'+file)
                    else:
                        paths.append(root+'/'+file)
        random.seed(42)
        random.shuffle(paths)
        return paths[0:self.len_limitation]
    
    def _adaptivepatch(self,img):
        w,h = img.size
        if min(w,h) < self.patchsize:
            img = torchvision.transforms.Resize((self.patchsize,self.patchsize))(img)
        else:
            w = w // self.patchsize * self.patchsize
            h = h // self.patchsize * self.patchsize
            img = torchvision.transforms.CenterCrop((w,h))(img)
        img = self.totensor(img)
        patches = img.unfold(1, self.patchsize, self.patchsize).unfold(2, self.patchsize, self.patchsize)
        num_patches = patches.shape[1] * patches.shape[2]
        patches = patches.contiguous().view(3, num_patches, self.patchsize, self.patchsize)
        patches = patches.permute(1,0,2,3)
        a,b,c,d = patches.shape
        if a > 19200:
            patches = patches[0:19200,:,:,:]
        return patches

    def _singlepatch(self,img):
        img = torchvision.transforms.Resize((224,224))(img)
        img = self.totensor(img)
        return img
    
    def __getitem__(self,index):
        path = self.paths[index]
        img = Image.open(path).convert('RGB')
        ## add noise
        if self.eval_noise == 'None':
            img = img
        elif self.eval_noise == 'jpg':
            img = pil_jpg_eval(img,int(90))
        elif self.eval_noise == 'resize':
            height,width=img.height,img.width
            img = torchvision.transforms.Resize((int(height*0.5),int(width*0.5)))(img)
        elif self.eval_noise == 'blur':
            img = np.array(img)
            gaussian_blur(img, 1.0)
            img = Image.fromarray(img)
        img_crops = self._adaptivepatch(img)
        return img_crops

    def __len__(self):
        return len(self.paths)
    

def save_list_to_file(my_list, filename):
    with open(filename, 'w') as file:
        json.dump(my_list, file)


def read_list_from_file(filename):
    with open(filename, 'r') as file:
        my_list = json.load(file)
    return my_list