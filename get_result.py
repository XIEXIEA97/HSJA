from resnet20 import ResNet20
from train_util import train, test
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
from train_util import train, test
from hsj_attack import HSJA
import random
import time

device = 'cuda' if torch.cuda.is_available() else 'cpu'

batch_size = 1
total_test = 1000
save_name = 'result'
net_name = 'proper_trained_net.pt'

# net = ResNet20().to(device)
# train(net, epochs=100, batch_size=100, lr=1e-1, reg=1e-4)

net = ResNet20().to(device)
net.load_state_dict(torch.load(net_name))
test(net)

transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

class model_wrapper():
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.count = 0

    def __call__(self, input):
        assert len(input.shape) == 4
        if isinstance(input, np.ndarray):
            res = self.model(torch.from_numpy(input).float().to(device))
        else:
            res = self.model(input.to(device))
        _, predicted = res.max(1)
        self.count += input.shape[0]
        return predicted.cpu().numpy()
    
    def count_reset(self):
        self.count = 0
    
    def get_count(self):
        return self.count
    
model = model_wrapper(net)

result = []
cat_count = [0 for i in range(10)]
test_count = 0
start_time = time.time()
hist = [0 for i in range(10)]
orig = [0 for i in range(10)]
with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(testloader):
        cate = targets[0]
        if cat_count[cate] >= total_test / 10:
            continue
        img = np.minimum(np.maximum(0, inputs), 1) 
        clean_predict = model(img)
        if clean_predict[0] != cate:
            continue
        model.count_reset()
        attack = HSJA(model, img, iterations = 1000000)
        if cat_count[cate] == 0:
            adv_img = torch.from_numpy(attack.generate(max_query = 25000, history = True)).float()
            orig[cate] = img
            hist[cate] = attack.get_histories()
        else:
            adv_img = torch.from_numpy(attack.generate(max_query = 25000)).float()            
        dss = attack.get_distances()
        adv_predict = model(adv_img)
        result.append(list(dss.items()))
        test_count += 1
        cat_count[cate] += 1
        if test_count % (total_test/100) == 0 or test_count % (total_test/10) == 0:
            print("[%d/%d] Time Taken: %d sec."%(test_count, total_test, time.time() - start_time))
        if test_count == total_test:
            break

result = np.array(result)
with open(save_name + '_data.npy', 'wb') as f:
    np.save(f, result)

# make trajectory
thres = [100, 200, 500, 1000, 2000, 5000, 10000, 25000]
imgs = [[0 for i in range(len(thres))] for j in range(len(hist))]
for i in range(len(hist)):
    idx = 0
    for query, img in hist[i].items():
        if query < thres[idx]:
            imgs[i][idx] = img
            if idx < len(thres)-1:
                imgs[i][idx+1] = img
        else:
            idx += 1
            if idx == len(thres):
                break
    imgs[i].insert(0, hist[i][0])
    imgs[i].append(orig[i])
    
imgs = np.array(imgs)
with open(save_name + '_trajectory.npy', 'wb') as f:
    np.save(f, imgs)