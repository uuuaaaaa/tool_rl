from gmm_fun import *
from config import opt



base_tag = 'basic_genimage'
eval_noise = opt.eval_noise
rate = opt.rate


TrainFeatures_base = torch.load('./features/train/{}.pth'.format(base_tag))
threshold_base = val_GMM_sklearn(modelpath='./checkpoints_gmm/{}'.format(base_tag), features_baseline= TrainFeatures_base,rate=rate)



val_gen = ['BigGAN',
        'ADM','Glide','Midjourney','stable_diffusion_v_1_4','stable_diffusion_v_1_5','VQDM','wukong'
           ]


record_t_map,record_t_acc = [],[]
for val_name in val_gen:
    
    loglikelihood_fake_base = read_list_from_file('./likelihood/genimage/fake_{}_{}'.format(base_tag, val_name))
    loglikelihood_real_base = read_list_from_file('./likelihood/genimage/real_{}_{}'.format(base_tag, val_name))



    acc, map = compute_mAP_acc_basic(val_name,threshold_base,loglikelihood_real_base,loglikelihood_fake_base)
    record_t_map.append(map)
    record_t_acc.append(acc)

print('Average | Tot: mAP:{:.2f} Acc:{:.2f} '.format(
    np.mean(record_t_map), np.mean(record_t_acc)
))