CUDA_VISIBLE_DEVICES=6,7 \
python tools/tsdf_fusion/generate_gt.py \
--test \
--data_path ../data/DATAROOT_SCANNET/scannet \
--save_name all_tsdf_9 \
--window_size 9 \
--n_proc 8 \
