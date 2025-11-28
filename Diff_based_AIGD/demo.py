from .gmm_fun import *
from .utils import *

def preprocessing(img):
    patchsize = 16
    totensor = torchvision.transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
        ])
    w,h = img.size
    if min(w,h) < patchsize:
        img = torchvision.transforms.Resize((patchsize,patchsize))(img)
    else:
        w = w // patchsize * patchsize
        h = h // patchsize * patchsize
        img = torchvision.transforms.CenterCrop((w,h))(img)
    img = totensor(img)
    patches = img.unfold(1, patchsize, patchsize).unfold(2, patchsize, patchsize)
    num_patches = patches.shape[1] * patches.shape[2]
    patches = patches.contiguous().view(3, num_patches, patchsize, patchsize)
    patches = patches.permute(1,0,2,3)
    a,b,c,d = patches.shape
    if a > 19200:
        patches = patches[0:19200,:,:,:]
    return patches
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)
def evaluate_image(image_path):
    
    net = load_net(checkpoint_path='./Diff_based_AIGD/checkpoints_backbone/basic.pth') 
    with open('./Diff_based_AIGD/checkpoints_gmm/basic_genimage', 'rb') as file:
        gmm = pickle.load(file)
    
    TrainFeatures_base = torch.load('./Diff_based_AIGD/features/train/basic_genimage.pth')
    threshold = val_GMM_sklearn(modelpath='./Diff_based_AIGD/checkpoints_gmm/basic_genimage', features_baseline= TrainFeatures_base,rate=0.0005)


    img = Image.open(image_path)
    img = preprocessing(img).cuda(device)
    f = net(img.unsqueeze(0))
    likelihood = gmm.score_samples(f.detach().cpu())
    print('log-likelihood:{}'.format(likelihood))
    if likelihood < threshold:
        result=1
        print('AI-generated')
    else:
        result=0
        print('Photographic')
      

    return result


