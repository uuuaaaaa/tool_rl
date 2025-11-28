from gmm_fun import *
from config import opt


bc_tag = 'ft_drct2m' 
base_tag = 'basic_drct2m'
eval_noise = 'None'

rate = opt.rate

def save_list_to_file(my_list, filename):
    with open(filename, 'w') as file:
        json.dump(my_list, file)


def read_list_from_file(filename):
    with open(filename, 'r') as file:
        my_list = json.load(file)
    return my_list



TrainFeatures_bc = torch.load('./features/train/{}.pth'.format(bc_tag))
threshold_bc = val_GMM_sklearn(modelpath='./checkpoints_gmm/{}'.format(bc_tag), features_baseline= TrainFeatures_bc,rate=rate)



TrainFeatures_base = torch.load('./features/train/{}.pth'.format(base_tag))
threshold_base = val_GMM_sklearn(modelpath='./checkpoints_gmm/{}'.format(base_tag), features_baseline= TrainFeatures_base,rate=rate)


loglikelihood_real_bc = read_list_from_file('./likelihood/drct2m/real_ft_drct2m_mscoco')
loglikelihood_real_base = read_list_from_file('./likelihood/drct2m/real_basic_drct2m_mscoco')

val_gen = ['stable-diffusion-inpainting','stable-diffusion-2-inpainting','stable-diffusion-xl-1.0-inpainting-0.1',
            'sd-controlnet-canny','sd21-controlnet-canny', 'controlnet-canny-sdxl-1.0',
           'ldm-text2im-large-256','stable-diffusion-v1-4','stable-diffusion-v1-5','stable-diffusion-2-1','stable-diffusion-xl-base-1.0',
           'stable-diffusion-xl-refiner-1.0','sd-turbo','sdxl-turbo','lcm-lora-sdv1-5','lcm-lora-sdxl',
           ]

record_t_map,record_t_acc = [],[]
for val_name in val_gen:
    loglikelihood_fake_base = read_list_from_file('./likelihood/drct2m/fake_basic_drct2m_{}'.format(val_name))
    loglikelihood_fake_bc = read_list_from_file('./likelihood/drct2m/fake_ft_drct2m_{}'.format(val_name))

    likelihood_real = [a + b for a, b in zip(loglikelihood_real_bc, loglikelihood_real_base)]  
    likelihood_fake = [a + b for a, b in zip(loglikelihood_fake_bc, loglikelihood_fake_base)]  

    acc_fusing, map_fusing = compute_mAP_acc_fusing(val_name,threshold_bc,threshold_base,loglikelihood_real_bc,loglikelihood_fake_bc,loglikelihood_real_base,loglikelihood_fake_base)
    record_t_map.append(map_fusing)
    record_t_acc.append(acc_fusing)

print('Average | Tot: mAP:{:.2f} Acc:{:.2f} '.format(
    np.mean(record_t_map), np.mean(record_t_acc)
))