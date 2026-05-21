import torch
import os
import lpips
import numpy as np
from tqdm import tqdm
from models import Gen
from data import TestDataset
from torch.utils.data import DataLoader
from config import DATA_ROOT
from torchvision.utils import save_image
from torchvision import transforms

def calculate_metrics(real_imgs, fake_imgs, device, lpips_model):
    """
    计算CycleGAN生成效果的客观指标
    1. LPIPS：感知相似度（越小越像，越逼真）
    2. MSE（简化FID）：像素级误差（越小越接近）
    """
    # 归一化到 [-1,1]
    real = (real_imgs - 0.5) * 2
    fake = (fake_imgs - 0.5) * 2

    # 1. LPIPS 感知损失（核心评价指标）
    lpips_score = lpips_model(real, fake).mean().item()

    # 2. MSE 像素误差（简单直观）
    mse_score = torch.mean((real - fake) ** 2).item()

    return lpips_score, mse_score

def infer(model_path, source_test, save_dir, task_name):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(save_dir, exist_ok=True)

    G = Gen().to(device)
    G.load_state_dict(torch.load(model_path, map_location=device))
    G.eval()


    dataset = TestDataset(root=DATA_ROOT, test_folder=source_test)
    loader = DataLoader(dataset, batch_size=1, num_workers=0)

    lpips_model = lpips.LPIPS(net='alex').to(device)
    all_lpips = []
    all_mse = []

    print(f"\n===== 开始推理：{task_name} =====")
    with torch.no_grad():
        for i, img in enumerate(tqdm(loader)):
            if i >= 500:  # 可改数量
                break

            img = img.to(device)
            fake_img = G(img)

            save_image(img, os.path.join(save_dir, f"{i+1}_real.png"), normalize=True)
            save_image(fake_img, os.path.join(save_dir, f"{i+1}_fake.png"), normalize=True)


            lp, mse = calculate_metrics(img, fake_img, device, lpips_model)
            all_lpips.append(lp)
            all_mse.append(mse)


    avg_lpips = np.mean(all_lpips)
    avg_mse = np.mean(all_mse)
    print(f"\n【{task_name} 平均指标】")
    print(f"LPIPS 感知相似度：{avg_lpips:.4f}（越小 → 越逼真）")
    print(f"MSE 像素误差：{avg_mse:.6f}（越小 → 越接近）")
    return avg_lpips, avg_mse

if __name__ == '__main__':
    # 1. 真实照片 → 莫奈风格 (B→A)
    infer(
        model_path="./model/gen_b2a.pth",
        source_test="testB",
        save_dir="./results/photo2monet",
        task_name="照片转莫奈"
    )

    # 2. 莫奈风格 → 真实照片 (A→B)
    infer(
        model_path="./model/gen_a2b.pth",  # 你需要有这个模型
        source_test="testA",
        save_dir="./results/monet2photo",
        task_name="莫奈转照片"
    )

