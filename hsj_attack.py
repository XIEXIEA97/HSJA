import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
# Implement the L2 version of HSJA and reproduce the untargeted attack

class HSJA():
    def __init__(self, model, sample, init_batch = 1e2, max_batch = 1e4, iterations = 40):
        self.model = model
        self.sample = sample.cpu().numpy()
        self.init_batch = init_batch
        self.max_batch = max_batch
        self.iterations = iterations
        self.real_label = model(sample)
        self.shape = sample.shape
        
        self.model.count_reset()
        self.start_count = self.model.get_count()

        # set theta
        self.d = np.prod(sample.shape)
        self.theta = np.power(self.d, -1.5)

        # initialize at x0, using blended uniform noise
        max_try = int(1e4)
        success = False
        for i in range(max_try):
            noise = np.random.uniform(0, 1, self.shape)
            success = self.phi(noise)
            if success:
                break
        assert success
        # low = 0.0
        # up = 1.0
        # while up - low > self.theta:
        #     mid = (low + up) / 2
        #     blended = (1 - mid) * sample + mid * noise
        #     if self.phi(self.model, blended):
        #         up = mid
        #     else:
        #         low = mid
        # self.x0 = (1 - up) * sample + up * noise
        # self.x0 = self.binary_search(noise)
        self.t_x0 = noise


    def clip(self, img):
        # keep pixel values in [0, 1]
        return np.minimum(np.maximum(0, img), 1) 
    
    def get_distance(self, x):
        return np.linalg.norm(x - self.sample)

    def phi(self, img):
        img = self.clip(img)
        result = self.model(img)
        return result != self.real_label
    
    def PI(self, alpha, x):
        return (1.0 - alpha) * self.sample + alpha * x

    def bin_search(self, x_):
        low = 0.0
        up = 1.0
        while up - low > self.theta:
            mid = (up + low) / 2
            if self.phi(self.PI(mid, x_)):
                up = mid
            else:
                low = mid
        return self.PI(up, x_)

    def monte_carlo_gradient(self, cur, batch, delta):
        # from original code
        shape = [batch] + list(self.shape[1:])
        u_b = np.random.randn(*shape)
        u_b = u_b / np.sqrt(np.sum(u_b ** 2, axis = (1,2,3), keepdims = True))

        perturbed = cur + delta * u_b
        perturbed = self.clip(perturbed)
        u_b = (perturbed - cur) / delta
        
        # print(cur.shape, u_b.shape, perturbed.shape) # (1, 3, 32, 32) (100, 3, 32, 32) (100, 3, 32, 32)

        # query the model.
        decisions = self.phi(perturbed)
        decision_shape = [len(decisions)] + [1] * len(self.shape[1:])
        fval = 2 * decisions.astype(float).reshape(decision_shape) - 1.0
        # print(fval.shape) # (100, 1, 1, 1, 1)

        # Baseline subtraction (when fval differs)
        if np.mean(fval) == 1.0: # label changes. 
            gradf = np.mean(u_b, axis = 0)
        elif np.mean(fval) == -1.0: # label not change.
            gradf = - np.mean(u_b, axis = 0)
        else:
            fval -= np.mean(fval)
#             print(fval.shape, u_b.shape)
            gradf = np.mean(fval * u_b, axis = 0) 

        # Get the gradient direction.
        gradf = gradf / np.linalg.norm(gradf)

        return gradf

    def generate(self, max_query, history = False):
        # compute d0
#         self.distance = [self.get_distance(self.t_x0)]
        self.distances = {}
        t_x = self.t_x0
        if history:
            self.histories = {0: t_x}
        for it in range(self.iterations):
            # boundary search
            x_t = self.bin_search(t_x)
            if self.model.get_count() - self.start_count > max_query:
                return self.clip(prev_x_t)
            prev_x_t = x_t
            self.distances[self.model.get_count() - self.start_count] = self.get_distance(prev_x_t)
            if history:
                self.histories[self.model.get_count() - self.start_count] = prev_x_t

            # gradient-direction estimation
            batch = int(min(self.init_batch * np.sqrt(it + 1), self.max_batch))
            delta = np.power(1.0 * self.d, -1) * self.get_distance(t_x)
            grad = self.monte_carlo_gradient(x_t, batch, delta)
#             print("grad shape: ",  grad.shape)

            # step size search
            ksi = 1.0 * self.get_distance(x_t) / np.sqrt(it + 1)
            while not self.phi(x_t + grad * ksi):
                ksi /= 2.0

            # Update the sample. 
            t_x = self.clip(x_t + grad * ksi)
#             self.distance.append(self.get_distance(t_x))            

        return self.clip(self.bin_search(t_x))
    
    def get_distances(self):
        return self.distances
    
    
    def get_histories(self):
        return self.histories