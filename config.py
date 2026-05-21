import os

# 自动创建输出文件夹
os.makedirs('./output/pics/', exist_ok=True)
os.makedirs('./output/pics_A/', exist_ok=True)
os.makedirs('./output/pics_B/', exist_ok=True)
os.makedirs('./model/', exist_ok=True)
os.makedirs('./model_bkp/', exist_ok=True)

DATA_ROOT = './true_data' 

class CFG:
    batch_size = 1
    image_size = 256
    use_gpu = True
    shuffle = True
