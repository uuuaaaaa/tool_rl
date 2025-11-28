from gmm_fun import *
from config import opt


bc_tag = 'ft_genimage' ## diffusion_bc_wd_70_noise
base_tag = 'basic_genimage'
eval_noise = 'None'

rate = opt.rate



TrainFeatures_bc = torch.load('./features/train/{}.pth'.format(bc_tag))
threshold_bc = val_GMM_sklearn(modelpath='./checkpoints_gmm/{}'.format(bc_tag), features_baseline= TrainFeatures_bc,rate=rate)



TrainFeatures_base = torch.load('./features/train/{}.pth'.format(base_tag))
threshold_base = val_GMM_sklearn(modelpath='./checkpoints_gmm/{}'.format(base_tag), features_baseline= TrainFeatures_base,rate=rate)



val_gen = ['BigGAN',
        'ADM','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong'
           ]


record_t_map,record_t_acc = [],[]
for val_name in val_gen:
    
    loglikelihood_fake_base = read_list_from_file('./likelihood/genimage/fake_basic_genimage_{}'.format(val_name))
    loglikelihood_fake_bc = read_list_from_file('./likelihood/genimage/fake_ft_genimage_{}'.format(val_name))

    loglikelihood_real_bc = read_list_from_file('./likelihood/genimage/real_ft_genimage_{}'.format(val_name))
    loglikelihood_real_base = read_list_from_file('./likelihood/genimage/real_basic_genimage_{}'.format(val_name))

    likelihood_real = [a + b for a, b in zip(loglikelihood_real_bc, loglikelihood_real_base)]  
    likelihood_fake = [a + b for a, b in zip(loglikelihood_fake_bc, loglikelihood_fake_base)]  

    acc_fusing, map_fusing = compute_mAP_acc_fusing(val_name,threshold_bc,threshold_base,loglikelihood_real_bc,loglikelihood_fake_bc,loglikelihood_real_base,loglikelihood_fake_base)
    record_t_map.append(map_fusing)
    record_t_acc.append(acc_fusing)

print('Average | Tot: mAP:{:.2f} Acc:{:.2f} '.format(
    np.mean(record_t_map), np.mean(record_t_acc)
))