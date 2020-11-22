import torch
import numpy as np
import matplotlib.pyplot as plt
import random
import os

def table1(result):
    thres = [1000, 5000, 20000]
    dist = [[0 for i in range(3)] for j in range(len(result))]
    for i in range(len(result)):
        idx = 0
        for j in result[i]:
            if j[0] < thres[idx]:
                dist[i][idx] = j[1]
            else:
                idx += 1
                if idx == 3:
                    break
    dist = np.array(dist)
    median = np.quantile(dist, 0.5, axis = 0)
    print("median l2 distance at 1K queries is:\t%f\n" \
    "median l2 distance at 5K queries is:\t%f\n" \
    "median l2 distance at 20K queries is:\t%f\n"%(median[0], median[1], median[2]))
    return median

def median_distance_vs_num_queries(result):
    q1 = np.quantile(result, 0.25, axis = 0)
    q2 = np.quantile(result, 0.5, axis = 0)
    q3 = np.quantile(result, 0.75, axis = 0)
    q4 = np.quantile(result, 0.99, axis = 0)

    l1, = plt.plot(q1[:, 0], q1[:, 1])
    l2, = plt.plot(q2[:, 0], q2[:, 1])
    l3, = plt.plot(q3[:, 0], q3[:, 1])
    l4, = plt.plot(q4[:, 0], q4[:, 1])
    plt.yscale("log")
    plt.ylim(1e-2, 1e1)
    plt.legend([l1, l2, l3], ['first quartiles', 'median', 'third quartiles'], loc='best')
    plt.legend([l1, l2, l3, l4], ['first quartiles', 'median', 'third quartiles', '99 percent'], loc='best')
    plt.ylabel('l2 distance')
    plt.xlabel('number of queries')
    plt.title("l2 distance vs. number of queries")
    plt.savefig("./visuals/distance_vs_queries.png")

def show_single_trajectory(imgs):         
    def show(img, i):
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.subplot(1, 10, i)
        plt.imshow(img)
        plt.axis("off")
        
    plt.figure(figsize=(50, 5))
    idx = random.randint(0, 9) 
    for j in range(10):
        show(imgs[idx][j][0], j + 1)
    plt.savefig("./visuals/single_trajectory")
    
def show_trajectories(imgs):         
    def show(img, i, j):
        img = np.transpose(img, (1, 2, 0))
        img = np.clip(img, 0, 1)
        plt.subplot(10, 10, 10 * i + j)
        plt.imshow(img)
        plt.axis("off")
        
    plt.figure(figsize=(50, 50))
    for i in range(10):
        for j in range(10):
            show(imgs[i][j][0], i, j + 1)
    plt.savefig("./visuals/trajectories")
    
if not os.path.exists('visuals'):
    os.makedirs('visuals')

with open('result_data.npy', 'rb') as f:
    tmp = np.load(f)
    median_distance_vs_num_queries(tmp)
    table1(tmp)

with open('result_trajectory.npy', 'rb') as f:
    tmp = np.load(f, allow_pickle=True)
    show_single_trajectory(tmp)
    show_trajectories(tmp)