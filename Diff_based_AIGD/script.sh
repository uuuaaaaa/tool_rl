python gmm_train_genimage.py --tag basic_genimage --checkpoint_path ./checkpoints_backbone/basic.pth --train_set_path /localssd/genimage_train --test_set_path /localssd/genimage_val 
python gmm_train_genimage.py --tag ft_genimage --checkpoint_path ./checkpoints_backbone/ft_genimage.pth --train_set_path /localssd/genimage_train --test_set_path /localssd/genimage_val 
python gmm_train_drct2m.py --tag basic_drct2m --checkpoint_path ./checkpoints_backbone/basic.pth --train_set_path /localssd/diffusion_2m_train/train_data --test_set_path /localssd/diffusion_2m_val
python gmm_train_drct2m.py --tag ft_drct2m --checkpoint_path ./checkpoints_backbone/ft_drct2m.pth --train_set_path /localssd/diffusion_2m_train/train_data --test_set_path /localssd/diffusion_2m_val

python eval_basic_genimage.py
python eval_basic_drct2m.py
python eval_fusing_genimage.py
python eval_fusing_drct2m.py