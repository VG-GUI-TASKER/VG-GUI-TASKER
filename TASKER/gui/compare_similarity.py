"""计算两组图片的相似度（SSIM、MSE、直方图相关性）- 仅依赖 opencv 和 numpy"""
import cv2
import numpy as np


def compute_ssim(img1, img2, C1=6.5025, C2=58.5225):
    """纯 numpy/opencv 实现的 SSIM"""
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float64)

    mu1 = cv2.GaussianBlur(gray1, (11, 11), 1.5)
    mu2 = cv2.GaussianBlur(gray2, (11, 11), 1.5)

    mu1_sq, mu2_sq, mu1_mu2 = mu1**2, mu2**2, mu1 * mu2
    sigma1_sq = cv2.GaussianBlur(gray1**2, (11, 11), 1.5) - mu1_sq
    sigma2_sq = cv2.GaussianBlur(gray2**2, (11, 11), 1.5) - mu2_sq
    sigma12 = cv2.GaussianBlur(gray1 * gray2, (11, 11), 1.5) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())


def compute_similarity(img_path1, img_path2):
    img1 = cv2.imread(img_path1)
    img2 = cv2.imread(img_path2)
    if img1 is None or img2 is None:
        raise FileNotFoundError(f"无法读取图片: {img_path1} 或 {img_path2}")

    # 统一尺寸
    h = min(img1.shape[0], img2.shape[0])
    w = min(img1.shape[1], img2.shape[1])
    img1 = cv2.resize(img1, (w, h))
    img2 = cv2.resize(img2, (w, h))

    # 1. SSIM
    ssim_val = compute_ssim(img1, img2)

    # 2. MSE
    mse_val = np.mean((img1.astype(float) - img2.astype(float)) ** 2)

    # 3. 直方图相关性
    hist_scores = []
    for i in range(3):
        h1 = cv2.calcHist([img1], [i], None, [256], [0, 256])
        h2 = cv2.calcHist([img2], [i], None, [256], [0, 256])
        cv2.normalize(h1, h1)
        cv2.normalize(h2, h2)
        hist_scores.append(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))
    hist_corr = np.mean(hist_scores)

    return {"ssim": ssim_val, "mse": mse_val, "hist_corr": hist_corr}


if __name__ == "__main__":
    import sys
    # Usage: python compare_similarity.py <image1> <image2>
    if len(sys.argv) >= 3:
        pairs = [(sys.argv[1], sys.argv[2])]
    else:
        base = "./MONDAY/images/tasker"
        pairs = [
            (f"{base}/_7Br1RTKU2c/frame_tasker_0004.png",
             f"{base}/_7Br1RTKU2c/frame_tasker_0005.png")
        ]

    for p1, p2 in pairs:
        res = compute_similarity(p1, p2)
        print(f"\n--- comparison ---")
        print(f"  image 1: {p1}")
        print(f"  image 2: {p2}")
        print(f"  SSIM:      {res['ssim']:.4f}  (closer to 1 = more similar)")
        print(f"  MSE:       {res['mse']:.2f}  (smaller = more similar)")
        print(f"  hist corr: {res['hist_corr']:.4f}  (closer to 1 = more similar)")
