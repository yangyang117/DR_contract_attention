import os
import torchvision.transforms as transforms

n_class = 5
epoches = 300
start_epoches = 24
lr= 7 * 1e-2
momentum=0.9
weight_decay=1e-4
opt_time = 6864 * 100
opt_min_lr = 7 * 1e-2
max_mo = 0.9
train_file = '/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/contract_train_over.csv'
test_file = '/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/contract_test.csv'
val_file = '/home/yyang2/data/yyang2/Data/EyeQ-master/results/contract/contract_val.csv'
start_epoches = 0
epoches = 100


