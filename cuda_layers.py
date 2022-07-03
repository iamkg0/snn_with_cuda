import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd

class Input_layer:
    def __init__(self, size=28*28, resolution=.1, preset=None, a=.02, b=.2, c=-65, d=2, learning_rate=.001, tau=20,
                    assymetry=1.05, g=20, inhibitory=False, update_weights=True, device=None):
        
        preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', None]
        param_list = [[0.02, 0.2, -65, 8],
						[0.02, 0.2, -55, 4],
						[0.02, 0.2, -50, 2],
						[0.1, 0.2, -65, 2],
						[0.02, 0.25, -65, 0.05],
						[0.1, 0.3, -65, 2],
					    [0.02, 0.25, -65, 2],
						[a, b, c, d]]

        assert preset in preset_list, f'Preset {preset} does not exist! Use one from {preset_list}'
        idx = preset_list.index(preset)
        self.a = param_list[idx][0]
        self.b = param_list[idx][1]
        self.c = param_list[idx][2]
        self.d = param_list[idx][3]
        
        self.I = torch.zeros(size).type(torch.FloatTensor)
        self.weights = None
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.impulse = torch.zeros(size).type(torch.FloatTensor)
        self.fire = self.impulse

        self.v = torch.ones(size).type(torch.FloatTensor) * self.c
        self.u = torch.ones(size).type(torch.FloatTensor) * b*self.v

        self.resolution = resolution
        self.learning_rate = learning_rate
        self.tau = tau / resolution
        self.assymetry = assymetry
        self.g = g
        self.update_weights = update_weights
        self.next_layer = None

        self.time = torch.zeros(size).type(torch.FloatTensor)
        self.spike_dt = torch.zeros(size).type(torch.FloatTensor)
        self.spike_t = torch.zeros(size).type(torch.FloatTensor)
        self.size = size
        self.device = device
        self.active_indices = []

        if device:
            self.I.to(device)
            self.impulse.to(device)
            self.v.to(device)
            self.u.to(device)

        self.defaults = self.v.clone(), self.u.clone()


    def __len__(self):
        return self.size


    def __getitem__(self, index):
        return




    def dynamics(self):
        self.v += self.resolution*(0.04*self.v**2 + 5*self.v + 140 - self.u + self.I)
        self.u += self.resolution*(self.a*(self.b * self.v - self.u))
        self.time += self.resolution
        self.spike_trace()


    def spike_trace(self):
        self.impulse -= self.impulse/self.tau


    def recovery(self):
        self.fire = self.impulse.clone()
        self.active_indices = []
        for i in range(len(self.I)):
            if self.v[i] >= 30:
                self.impulse[i] += 1
                self.fire[i] += 1
                self.v[i] = self.c
                self.u[i] += self.d
                self.spike_t[i] = self.time[i]
                self.spike_dt[i] = 0
                self.active_indices.append(i)
            else:
                self.fire[i] = 0
        return self.fire


    def apply_current(self, signal):
        sig = torch.tensor(signal)
        if self.device:
            sig.to(self.device)
        self.I = sig    

    
    def STDP(self):
        if self.update_weights == True:
            #DOES NOT WORK PROPERLY!!! REQUIRES REWORK!!!
            # or does...
            if self.active_indices:
                impulse_tensor = self.impulse[self.active_indices].repeat(len(self.next_layer)).reshape((len(self.active_indices), len(self.next_layer)))
                #print(impulse_tensor.shape, self.weights[self.active_indices,:].shape)
                self.weights[self.active_indices] -= self.learning_rate * self.weights[self.active_indices,:] * impulse_tensor
            if self.next_layer.active_indices:
                self.weights[:,self.next_layer.active_indices] = 1 - self.weights[:,self.next_layer.active_indices]
                plus = self.learning_rate * self.assymetry * self.weights[:,self.next_layer.active_indices] * self.next_layer.impulse[self.next_layer.active_indices]
                self.weights[:,self.next_layer.active_indices] += plus
            '''
            if self.weights.max() >= 1 or self.weights.min() <=0:
                lowest = self.weights.min() - 1e-3
                highest = self.weights.max() + 1e-3
                self.weights = (self.weights - lowest) / (highest - lowest)
            
            for i in range(self.weights.shape[0]):
                for j in range(self.weights.shape[1]):
                    if self.weights[i,j] <= 0:
                        self.weights[i,j] = 1e-3
                    if self.weights[i,j] >= 1:
                        self.weights[i,j] = 1 - .001
            '''

    def transmit_current(self):
        impulse, _ = self.behave()
        self.next_layer.I += torch.matmul(self.weights.clone().T, impulse.reshape((self.size, 1))).reshape(self.next_layer.size) * self.g


    def make_connections(self, layer):
        self.next_layer = layer
        self.weights = torch.rand((self.size, len(layer)))
        if self.device:
            self.weights.to(self.device)


    def behave(self):
        self.dynamics()
        if self.next_layer:
            self.STDP()
        impulse = self.recovery()
        return impulse, self.v


    def save_weights(self):
        raise Exception('Not implemented!')


    def load_weights(self):
        raise Exception('Not implemented!')

    
    def reboot_variables(self):
        self.I *= 0
        self.impulse *= 0
        self.v, self.u = self.defaults
        
        #raise Exception('Not implemented!')


    def normalize_weights(self):
        if self.weights.max() >= 1 or self.weights.min() <=0:
            lowest = self.weights.min() - 1e-3
            highest = self.weights.max() + 1e-3
            self.weights = (self.weights - lowest) / (highest - lowest)





class Output_layer:
    def __init__(self, size=28*28, resolution=.1, preset=None, a=.02, b=.2, c=-65, d=2, learning_rate=.001, tau=20,
                    assymetry=1.05, g=20, inhibitory=False, update_weights=True, device=None):
        
        preset_list = ['RS', 'IB', 'CH', 'FS', 'TC', 'RZ', 'LTS', None]
        param_list = [[0.02, 0.2, -65, 8],
						[0.02, 0.2, -55, 4],
						[0.02, 0.2, -50, 2],
						[0.1, 0.2, -65, 2],
						[0.02, 0.25, -65, 0.05],
						[0.1, 0.3, -65, 2],
					    [0.02, 0.25, -65, 2],
						[a, b, c, d]]

        assert preset in preset_list, f'Preset {preset} does not exist! Use one from {preset_list}'
        idx = preset_list.index(preset)
        self.a = param_list[idx][0]
        self.b = param_list[idx][1]
        self.c = param_list[idx][2]
        self.d = param_list[idx][3]
        
        self.I = torch.zeros(size).type(torch.FloatTensor)
        self.weights = None
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.impulse = torch.zeros(size).type(torch.FloatTensor)
        self.fire = self.impulse

        self.v = torch.ones(size).type(torch.FloatTensor) * self.c
        self.u = torch.ones(size).type(torch.FloatTensor) * b*self.v

        self.resolution = resolution
        self.learning_rate = learning_rate
        self.tau = tau / resolution
        self.assymetry = assymetry
        self.g = g
        self.update_weights = update_weights
        self.next_layer = None

        self.time = torch.zeros(size).type(torch.FloatTensor)
        self.spike_dt = torch.zeros(size).type(torch.FloatTensor)
        self.spike_t = torch.zeros(size).type(torch.FloatTensor)
        self.size = size
        self.device = device
        self.active_indices = []

        if device:
            self.I.to(device)
            self.impulse.to(device)
            self.v.to(device)
            self.u.to(device)

        self.defaults = self.v.clone(), self.u.clone()


    def __len__(self):
        return self.size


    def __getitem__(self, index):
        return


    def dynamics(self):
        self.v += self.resolution*(0.04*self.v**2 + 5*self.v + 140 - self.u + self.I)
        self.u += self.resolution*(self.a*(self.b * self.v - self.u))
        self.time += self.resolution
        self.spike_trace()


    def spike_trace(self):
        self.impulse -= self.impulse/self.tau


    def recovery(self):
        self.fire = self.impulse.clone()
        self.active_indices = []
        for i in range(len(self.I)):
            if self.v[i] >= 30:
                self.impulse[i] += 1
                self.fire[i] += 1
                self.v[i] = self.c
                self.u[i] += self.d
                self.spike_t[i] = self.time[i]
                self.spike_dt[i] = 0
                self.active_indices.append(i)
            else:
                self.fire[i] = 0
        return self.fire


    def apply_current(self, signal):
        sig = torch.tensor(signal)
        if self.device:
            sig.to(self.device)
        self.I = sig


    def drop_impulse(self):
        self.I *= 0


    def behave(self):
        self.dynamics()
        impulse = self.recovery()
        return impulse, self.v


    def reboot_variables(self):
        self.I *= 0
        self.impulse *= 0
        self.v, self.u = self.defaults