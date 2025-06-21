# CUDA_VISIBLE_DEVICES=4,5,6,7 nohup python run_nerf.py --config configs/colmap.txt > colmap_multi_gpu.log 2>&1 &
# CUDA_VISIBLE_DEVICES=7 nohup python run_nerf.py --config configs/colmap.txt > colmap_SINGLE_gpu.log 2>&1 &
CUDA_VISIBLE_DEVICES=5 nohup python run_nerf.py --config configs/colmap.txt > psnr_ssim.log 2>&1 &