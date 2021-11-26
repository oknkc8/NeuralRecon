PYOPENGL_PLATFORM=osmesa \
python tools/evaluation.py \
--model ./results/ \
--data_path ../data/DATAROOT_SCANNET/scannet/scans_test \
--gt_path ../data/DATAROOT_SCANNET/scannet/scans_test \
--n_proc 8 \
