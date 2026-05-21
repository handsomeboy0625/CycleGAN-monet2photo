import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import random
import matplotlib.pyplot as plt 
from models import Gen, Disc
from utils import show_pics, save_pics, ImagePool, open_pic
from data import PairDataset
from torch.utils.data import DataLoader
from config import DATA_ROOT

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    G_A2B = Gen().to(device)
    G_B2A = Gen().to(device)
    D_A = Disc().to(device)
    D_B = Disc().to(device)

    opt_G = optim.Adam(list(G_A2B.parameters()) + list(G_B2A.parameters()), lr=0.0002, betas=(0.5, 0.999))
    opt_D = optim.Adam(list(D_A.parameters()) + list(D_B.parameters()), lr=0.0002, betas=(0.5, 0.999))


    pool_A = ImagePool()
    pool_B = ImagePool()

    dataset = PairDataset(DATA_ROOT)
    loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)


    num_epochs = 120
    step = 0
=
    loss_D_list = []
    loss_G_list = []
    step_list = []

    print("=== Start Training (Epoch Mode) ===")

    for epoch in range(num_epochs):
        for real_A, real_B in loader:
            real_A = real_A.to(device)
            real_B = real_B.to(device)
            step += 1

            opt_D.zero_grad()

            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)

            loss_D_A = (torch.mean((D_A(real_A)-1)**2) + torch.mean(D_A(pool_A.query(fake_A.detach()))**2)) * 0.5
            loss_D_B = (torch.mean((D_B(real_B)-1)**2) + torch.mean(D_B(pool_B.query(fake_B.detach()))**2)) * 0.5
            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            opt_D.step()

            # ====================== 更新生成器 G ======================
            opt_G.zero_grad()

            fake_B = G_A2B(real_A)
            fake_A = G_B2A(real_B)
            rec_A = G_B2A(fake_B)
            rec_B = G_A2B(fake_A)

            loss_gan_A = torch.mean((D_A(fake_A)-1)**2)
            loss_gan_B = torch.mean((D_B(fake_B)-1)**2)
            loss_cycle = torch.mean(torch.abs(real_A - rec_A)) + torch.mean(torch.abs(real_B - rec_B))
            loss_idt = torch.mean(torch.abs(real_A - G_B2A(real_A))) + torch.mean(torch.abs(real_B - G_A2B(real_B)))

            loss_G = loss_gan_A + loss_gan_B + 30*loss_cycle + 10*loss_idt
            loss_G.backward()
            opt_G.step()

            if step % 50 == 0:
                loss_D_list.append(loss_D.item())
                loss_G_list.append(loss_G.item())
                step_list.append(step)

            if step % 100 == 0:
                print(f"Epoch: [{epoch+1}/{num_epochs}] | Step: {step} | D: {loss_D.item():.3f} | G: {loss_G.item():.1f}")

            if step % 1000 == 0:
                save_pics([real_A, fake_B, real_B, fake_A], str(step))
                
                test_img_path = os.path.join(DATA_ROOT, "trainB", os.listdir(os.path.join(DATA_ROOT, "trainB"))[0])
                test_img = open_pic(test_img_path).to(device)
                save_pics([test_img, G_B2A(test_img)], str(step), './output/pics_B/')

                test_img_path = os.path.join(DATA_ROOT, "trainA", os.listdir(os.path.join(DATA_ROOT, "trainA"))[0])
                test_img = open_pic(test_img_path).to(device)
                save_pics([test_img, G_A2B(test_img)], str(step), './output/pics_A/')

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(step_list, loss_D_list, label="D Loss", color='blue')
    plt.xlabel("Step")
    plt.ylabel("Discriminator Loss")
    plt.title("D Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(step_list, loss_G_list, label="G Loss", color='red')
    plt.xlabel("Step")
    plt.ylabel("Generator Loss")
    plt.title("G Loss Curve")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("./loss_curve.png", dpi=300)  
    plt.show()

    # 保存模型
    torch.save(G_A2B.state_dict(), './model/gen_a2b.pth')
    torch.save(G_B2A.state_dict(), './model/gen_b2a.pth')
    print("=== Training Done! ===")

if __name__ == '__main__':
    train()
