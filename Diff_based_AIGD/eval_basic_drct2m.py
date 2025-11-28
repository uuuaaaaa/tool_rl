from gmm_fun import *
from config import opt



base_tag = 'basic_drct2m'
eval_noise = opt.eval_noise
rate = opt.rate





TrainFeatures_base = torch.load('./features/train/{}.pth'.format(base_tag))
threshold_base = val_GMM_sklearn(modelpath='./checkpoints_gmm/{}'.format(base_tag), features_baseline= TrainFeatures_base,rate=rate)

loglikelihood_real_base = read_list_from_file('./likelihood/drct2m/real_basic_drct2m_mscoco')

val_gen = ['stable-diffusion-inpainting','stable-diffusion-2-inpainting','stable-diffusion-xl-1.0-inpainting-0.1',
            'sd-controlnet-canny','sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0',
           'ldm-text2im-large-256','stable-diffusion-v1-4','stable-diffusion-v1-5','stable-diffusion-2-1','stable-diffusion-xl-base-1.0',
           'stable-diffusion-xl-refiner-1.0','sd-turbo','sdxl-turbo','lcm-lora-sdv1-5','lcm-lora-sdxl',
           ]

record_t_map,record_t_acc = [],[]
for val_name in val_gen:
    loglikelihood_fake_base = read_list_from_file('./likelihood/drct2m/fake_{}_{}'.format(base_tag,val_name))
    acc, map = compute_mAP_acc_basic(val_name,threshold_base,loglikelihood_real_base,loglikelihood_fake_base)
    record_t_map.append(map)
    record_t_acc.append(acc)


print('Average | Tot: mAP:{:.2f} Acc:{:.2f} '.format(
    np.mean(record_t_map), np.mean(record_t_acc)
))