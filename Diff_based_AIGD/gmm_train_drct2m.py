
from config import opt
from gmm_fun import *
torch.set_num_threads(12)


print('checkpoint_path:{} | noise:{}'.format(opt.checkpoint_path,opt.eval_noise))
net = load_net(checkpoint_path=opt.checkpoint_path)

# train set feature extraction & gmm training
TrainDataset_mscoco = DatasetforDiffusionGranularityBinary(root='{}/0_real/'.format(opt.train_set_path), filter_word = None, filter_mode=False, len_limitation = 10000, eval_noise = None)
TrainFeatures = img2f(net,TrainDataset_mscoco)
torch.save(TrainFeatures.cpu(), './features/train/{}.pth'.format(opt.tag))
train_GMM_sklearn(TrainFeatures,'./checkpoints_gmm/{}'.format(opt.tag))
# TrainFeatures = torch.load('./features/train/{}.pth'.format(opt.tag))


## test image feature extraction & gmm inference
val_gen = ['stable-diffusion-inpainting','stable-diffusion-2-inpainting','stable-diffusion-xl-1.0-inpainting-0.1',
            'sd-controlnet-canny','sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0',
           'ldm-text2im-large-256','stable-diffusion-v1-4','stable-diffusion-v1-5','stable-diffusion-2-1','stable-diffusion-xl-base-1.0',
           'stable-diffusion-xl-refiner-1.0','sd-turbo','sdxl-turbo','lcm-lora-sdv1-5','lcm-lora-sd'
           'xl',
           ]
TestDatasetReal = DatasetforDiffusionGranularityBinary(root='/localssd/mscoco_val/', filter_word = '0_real', filter_mode=False, len_limitation = 50000, eval_noise = opt.eval_noise)
TestRealFea = img2f(net,TestDatasetReal)
torch.save(TestRealFea.cpu(), './features/test/{}_{}_real.pth'.format(opt.tag, 'mscoco'))
# TestRealFea = torch.load('./features/test/{}_{}_real.pth'.format(opt.tag, val_name))
loglikelihood_real = f2loglikelihood('./checkpoints_gmm/{}'.format(opt.tag),TestRealFea)
save_list_to_file(loglikelihood_real,'./likelihood/drct2m/real_{}_{}'.format(opt.tag, 'mscoco'))

for val_name in val_gen:
    TestDatasetFake = DatasetforDiffusionGranularityBinary(root='{}/{}/'.format(opt.test_set_path,val_name), filter_word = '1_fake', filter_mode=False, len_limitation = 50000, eval_noise = opt.eval_noise)
    TestFakeFea = img2f(net,TestDatasetFake)
    torch.save(TestFakeFea.cpu(), './features/test/{}_{}_fake.pth'.format(opt.tag, val_name))
    # TestFakeFea = torch.load('./features/test/{}_{}_fake.pth'.format(opt.tag, val_name))
    loglikelihood_fake = f2loglikelihood('./checkpoints_gmm/{}'.format(opt.tag),TestFakeFea)
    save_list_to_file(loglikelihood_fake,'./likelihood/drct2m/fake_{}_{}'.format(opt.tag, val_name))