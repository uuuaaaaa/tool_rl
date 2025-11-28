
from config import opt
from gmm_fun import *
torch.set_num_threads(12)


print('checkpoint_path:{} | noise:{}'.format(opt.checkpoint_path,opt.eval_noise))
net = load_net(checkpoint_path=opt.checkpoint_path)

# train set feature extraction & gmm training
TrainDataset_imagenet = DatasetforDiffusionGranularityBinary(root='{}/0_real/'.format(opt.train_set_path), filter_word = None, filter_mode=False, len_limitation = 10000, eval_noise = None)
TrainFeatures = img2f(net,TrainDataset_imagenet)
torch.save(TrainFeatures.cpu(), './features/train/{}.pth'.format(opt.tag))
train_GMM_sklearn(TrainFeatures,'./checkpoints_gmm/{}'.format(opt.tag))
# TrainFeatures = torch.load('./features/train/{}.pth'.format(opt.tag))
threshold = val_GMM_sklearn(modelpath='./checkpoints_gmm/{}'.format(opt.tag), features_baseline= TrainFeatures, rate=opt.rate)

## test image feature extraction & gmm inference
val_gen = ['BigGAN', 'ADM','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong']
for val_name in val_gen:
    TestDatasetReal = DatasetforDiffusionGranularityBinary(root='{}/{}/0_real/'.format(opt.test_set_path,val_name), filter_word = '0_real', filter_mode=False, len_limitation = 50000, eval_noise = opt.eval_noise)
    TestDatasetFake = DatasetforDiffusionGranularityBinary(root='{}/{}/1_fake/'.format(opt.test_set_path,val_name), filter_word = '1_fake', filter_mode=False, len_limitation = 50000, eval_noise = opt.eval_noise)
    TestRealFea = img2f(net,TestDatasetReal)
    TestFakeFea = img2f(net,TestDatasetFake)
    torch.save(TestRealFea.cpu(), './features/test/{}_{}_real.pth'.format(opt.tag, val_name))
    torch.save(TestFakeFea.cpu(), './features/test/{}_{}_fake.pth'.format(opt.tag, val_name))
    # TestRealFea = torch.load('./features/test/{}_{}_real.pth'.format(opt.tag, val_name))
    # TestFakeFea = torch.load('./features/test/{}_{}_fake.pth'.format(opt.tag, val_name))
    loglikelihood_real = f2loglikelihood('./checkpoints_gmm/{}'.format(opt.tag),TestRealFea)
    loglikelihood_fake = f2loglikelihood('./checkpoints_gmm/{}'.format(opt.tag),TestFakeFea)
    save_list_to_file(loglikelihood_real,'./likelihood/genimage/real_{}_{}'.format(opt.tag, val_name))
    save_list_to_file(loglikelihood_fake,'./likelihood/genimage/fake_{}_{}'.format(opt.tag, val_name))