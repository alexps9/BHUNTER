# Appearance transfer

This directory adapts [Unsup_Recycle_GAN](https://github.com/wangkaihong/Unsup_Recycle_GAN) (AAAI 2022) to translate CARLA camera frames toward a real-image appearance. The pretrained weights used by BHUNTER are stored at:

```
style_transfer/checkpoints/bhunter_sim2real/
├── 50_net_G_A.pth
├── 50_net_G_B.pth
├── 50_net_D_A.pth
└── 50_net_D_B.pth
```

Run commands from this directory. Organize unpaired frames as:

```
path/to/data/
├── train/
│   ├── A/    # simulated frames
│   └── B/    # real frames
└── val/
    ├── A/
    └── B/
```

Translate a validation set with the released checkpoint:

```bash
python test.py \
  --dataroot path/to/data \
  --model unsup_single \
  --dataset_mode unaligned_scale \
  --name bhunter_sim2real \
  --which_epoch 50 \
  --loadSizeW 512 --loadSizeH 256 \
  --resize_mode rectangle \
  --fineSizeW 512 --fineSizeH 256 \
  --crop_mode none \
  --which_model_netG resnet_6blocks \
  --no_dropout \
  --gpu_ids 0
```

Results are written under `results/`. Training uses the same flags with `train.py` and `--niter` / `--niter_decay`. `--model reCycle_gan` selects the temporal ReCycle variant in `models/bhunter_recycle_model.py`.

Please also cite Unsup_Recycle_GAN when using this component. See `NOTICE` in the repository root.
