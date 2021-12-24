#!/bin/bash
python train.py \
--name horse_on_a_side_augment --batch 4 \
--dataroot_sketch ./data/sketch/photosketch/horse_on_a_side \
--dataroot_image ./data/image/horse --l_image 0.7 \
--g_pretrained ./pretrained/stylegan2-horse/netG.pth \
--d_pretrained ./pretrained/stylegan2-horse/netD.pth \
--disable_eval --diffaug_policy translation \

