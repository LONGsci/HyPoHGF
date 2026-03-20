import os
import sys

# 随机种子
seed = 2021

# 参数类
class para:
    def __init__(self, lr, rec, drop, batchSize, epoch, dev_ratio, test_ratio):
        self.lr = lr
        self.rec = rec
        self.drop = drop
        self.batchSize = batchSize
        self.epoch = epoch
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

# 超参数配置
params = para(lr=3e-4, rec=7e-3, drop=0.0, batchSize=256, epoch=300, dev_ratio=0.15, test_ratio=0.15)

# 路径/目录配置
checkpoint_dir = 'model_save'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()