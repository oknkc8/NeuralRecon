PYOPENGL_PLATFORM=osmesa \
python tools/evaluation.py \
--model ./results/legacy/0924/scene_scannet_test_model_fusion_eval_35 \
--data_path ../data/DATAROOT_SCANNET/scannet/scans_test \
--gt_path ../data/DATAROOT_SCANNET/scannet/scans_test \
--n_proc 8 \
