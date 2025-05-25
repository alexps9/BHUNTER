# Unsupervised Recycle-GAN
Code for [Learning Temporally and Semantically Consistent Unpaired Video-to-video Translation Through Pseudo-Supervision From Synthetic Optical Flow](https://arxiv.org/pdf/2201.05723.pdf), AAAI 2022

# Introduction

Our Unsupervised Recycle-GAN seeks to improve the quality of translation in the unpaired video-to-video translation task and address the temporal and semantic inconsistency issues leveraging pseudo-supervision from synthetic optical flow.

![Overall](figs/fig1.png)

Our Unsupervised Recycle-GAN includes a branch of unsupervised recycle loss and a branch of unsupervised spatial loss that provide more accurate and efficient spatiotemporal consistency constraints during the adversarial training. Built on the basis of an unpaired image-to-image translation framework (Cycle-GAN in this example), our method can effectively improve the temproal and semantic consistency in the resulting videos. 

![Losses](figs/fig2.png)

# Usage

**Data Preparation**

Viper dataset is available via [Recycle-GAN](https://github.com/aayushbansal/Recycle-GAN/), and CityScapes sequence dataset (leftImg8bit_sequence_trainvaltest) is available [Here](https://www.cityscapes-dataset.com/downloads/). Please be advised that preparing datasets in triplets (as in Recycle-GAN) is not neccessary unless you also need to run Recycle-GAN.

Organize the dataset in such a way that it contains train/val set and source domain A/ target domain B hierarchically. For Viper-to-CityScapes experiments, A/B will be the frames from Viper/CityScapes while for Video-to-Label experiments, A/B will be the frames/label maps in Viper. 
```
path/to/data/
|-- train
|   |-- A
|   `-- B
`-- val
    |-- A
    `-- B
```

**Viper-to-CityScapes Experiment**
```
python train.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 542 --loadSizeH 286 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 --lambda_spa_unsup_A 10 --lambda_spa_unsup_B 10 --lambda_unsup_cycle_A 10 --lambda_unsup_cycle_B 10 --lambda_cycle_A 0 --lambda_cycle_B 0 --lambda_content_A 1 --lambda_content_B 1 --batchSize 1 --noise_level 0.001  --niter_decay 0 --niter 2
python test.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2c_experiment --loadSizeW 512 --loadSizeH 256 --resize_mode rectangle --fineSizeW 512 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout --which_epoch 2
```

**Video-to-Label Experiment**

```
python train.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2l_experiment --loadSizeW 286 --loadSizeH 286 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 --crop_mode rectangle --which_model_netG resnet_6blocks --no_dropout --pool_size 0 --lambda_spa_unsup_A 10 --lambda_spa_unsup_B 0 --lambda_unsup_cycle_A 0 --lambda_unsup_cycle_B 10 --lambda_cycle_A 10 --lambda_cycle_B 10 --lambda_content_A 0 --lambda_content_B 0 --batchSize 1 --noise_level 0.001  --niter_decay 0 --niter 5
python test.py --dataroot path/to/data/ --model unsup_single --dataset_mode unaligned_scale --name v2l_experiment --loadSizeW 256 --loadSizeH 256 --resize_mode rectangle --fineSizeW 256 --fineSizeH 256 --crop_mode none --which_model_netG resnet_6blocks --no_dropout --which_epoch 5
```
   
**Pretrained Models**

Pretrained models in both experiments are available [here](https://www.dropbox.com/scl/fo/kcf2ycdvvg4o25k20k8xn/AK90oBpNgteB7ZELqoYlhbI?rlkey=2ae3o3cqypzodl55j1czz343v&st=grbm95jb&dl=0).

**Evaluation**

Pretrained FCN model is available [here](https://drive.google.com/file/d/1NmeC32gGoKqitxBax21-FaAKL_cBwwqg/view?usp=sharing). Please place it under .saved_models/.
```
python util/eval_rgb2lbl.py --exp_name path/to/test_output/images/\*fake_B.\* --map_cache_dir seg_map_cache
python util/eval_lbl2rgb.py --exp_name path/to/test_output/images/\*fake_A\* --pred_cache_dir pred_map_cache --mean False --model_path saved_models/fcn_model.pt
```

# Citation
```
@article{Wang_Akash_Misu_2022, 
title={Learning Temporally and Semantically Consistent Unpaired Video-to-video Translation Through Pseudo-Supervision From Synthetic Optical Flow}, 
journal={Proceedings of the AAAI Conference on Artificial Intelligence}, 
author={Wang, Kaihong and Akash, Kumar and Misu, Teruhisa}, 
year={2022}, 
}
```
             
**Acknowledgment**

Code borrowed from [Recycle-GAN](https://github.com/aayushbansal/Recycle-GAN/), and [ReReVST](https://github.com/daooshee/ReReVST-Code/). We especially thank Dr. Aayush Bansal for his helps!
