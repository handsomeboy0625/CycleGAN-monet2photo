# CycleGAN 风格迁移（照片 ↔ 莫奈）
极简可用版 README

项目结构：
├── README.md
├── requirement.txt   
├── train.py          模型训练
├── infer.py          推理风格迁移
├── models.py         生成器、判别器
├── data.py           数据集加载
├── utils.py          图片工具、ImagePool
├── config.py         路径与参数配置
├── model/            保存训练好的模型        
    ├── gen_a2b.pth    A->B
    ├── gen_a2b.pth    B->A
├── output/           训练效果图
├── results/          存放推断结果
├── loss_curve.png    loss曲线
└── true_data/         数据集（由于文件过大仅仅留下部分用于测试）
    ├── trainA/        风格A（莫奈）
    ├── trainB/        风格B（照片）
    ├── testA/         测试A
    └── testB/         测试B
## 各文件作用
train.py
  模型训练，更新G和D，保存模型，输出loss曲线和效果图

infer.py
  加载训练好的模型，对测试图进行风格迁移并显示

models.py
  定义生成器 Gen、判别器 Disc、残差块

data.py
  PairDataset：训练集加载
  TestDataset：测试集加载

utils.py
  图片加载、显示、保存；ImagePool 缓冲池

config.py
  配置数据集路径，自动创建输出文件夹

## 使用方法
进入项目根目录
训练：
python train.py

推理：
python infer.py


## 注意事项
- 首次运行会自动创建 ./output/、./model/ 等文件夹；
- 训练时间取决于硬件，建议使用 GPU 加速。