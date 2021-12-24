#!/bin/bash
python  train.py \
--name standing_cat_augment --batch 4 \
--dataroot_sketch ./data/sketch/photosketch/standing_cat \
--dataroot_image ./data/image/cat --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-cat/netG.pth \
--d_pretrained ./pretrained/stylegan2-cat/netD.pth \
--disable_eval --diffaug_policy translation \



