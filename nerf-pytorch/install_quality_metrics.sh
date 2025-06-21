#!/bin/bash

echo "Installing image quality assessment libraries..."

# 安装scikit-image (用于SSIM)
echo "Installing scikit-image..."
pip install scikit-image

# 安装LPIPS
echo "Installing LPIPS..."
pip install lpips

# 验证安装
echo "Verifying installations..."
python -c "
try:
    from skimage.metrics import structural_similarity, peak_signal_noise_ratio
    print('✓ scikit-image installed successfully')
except ImportError as e:
    print('✗ scikit-image installation failed:', e)

try:
    import lpips
    print('✓ LPIPS installed successfully')
except ImportError as e:
    print('✗ LPIPS installation failed:', e)

try:
    import cv2
    print('✓ OpenCV already available')
except ImportError:
    print('! OpenCV not available (optional)')
"

echo "Installation complete!"
echo ""
echo "Available metrics:"
echo "- PSNR: Peak Signal-to-Noise Ratio (higher is better)"
echo "- SSIM: Structural Similarity Index (0-1, higher is better)"
echo "- LPIPS: Learned Perceptual Image Patch Similarity (lower is better)" 