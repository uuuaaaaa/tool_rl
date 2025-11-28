import argparse
import os
import torch
import random
import numpy as np
import warnings 
warnings.filterwarnings("ignore")

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--tag', type=str, default='ft_genimage')
        parser.add_argument('--train_set_path', type=str, default='/localssd/genimage_train')
        parser.add_argument('--test_set_path', type=str, default='/localssd/genimage_val')
        parser.add_argument('--eval_mode', type=str, default='basic')
        parser.add_argument('--eval_noise', type=str, default='None')
        parser.add_argument('--eval_noise_param', type=float)
        parser.add_argument('--checkpoint_path', type=str, default='./checkpoints_backbone/ft_genimage.pth')
        parser.add_argument('--rate', type=float, default=0.0005)
        parser.add_argument('--seed', type=int, default=42)     
        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        opt, _ = parser.parse_known_args()
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        opt = self.gather_options()
        return opt
    
opt = BaseOptions().parse()


set_random_seed(opt.seed)
