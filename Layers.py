import numpy as np
import torch
import matplotlib.pyplot as plt

if torch.cuda.is_available() == True:
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

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
        
        self.I = torch.zeros(size)
        self.weights = None
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.impulse = torch.zeros(size)
        self.fire = self.impulse

        self.v = torch.ones(size) * self.c
        self.u = torch.ones(size) * b*self.v

        self.resolution = resolution
        self.learning_rate = learning_rate
        self.tau = tau / resolution
        self.assymetry = assymetry
        self.g = g
        self.update_weights = update_weights
        self.next_layer = None

        self.time = torch.zeros(size)
        self.spike_dt = torch.zeros(size)
        self.spike_t = torch.zeros(size)
        self.size = size
        self.device = device

        if device:
            self.I.to(device)
            self.impulse.to(device)
            self.v.to(device)
            self.u.to(device)




    def dynamics(self):
        self.v += self.resolution*(0.04*self.v**2 + 5*self.v + 140 - self.u + self.I)
        self.u += self.resolution*(self.a*(self.b * self.v - self.u))
        self.time += self.resolution
        self.spike_trace()


    def spike_trace(self):
        self.impulse -= self.impulse/self.tau


    def recovery(self):
        self.fire = self.impulse
        for i in range(len(self.I)):
            if self.v[i] >= 30:
                self.fire[i] += 1
                self.v[i] = self.c[i]
                self.u[i] += self.d
                self.spike_t[i] = self.time[i]
                self.spike_dt[i] = 0
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
            pass


    def transmit_current(self):
        impulse, _ = self.behave()
        self.next_layer.I += torch.matmul(self.weights, impulse.rehsape((self.size, 1))).reshape(self.next_layer.size)


    def make_connections(self, layer):
        self.next_layer = layer
        self.weights = torch.rand((self.size, len(layer)))
        if self.device:
            self.weights.to(device)


    def behave(self):
        self.dynamics()
        self.STDP()
        impulse = self.recovery()
        return impulse, self.v



class Output_layer:
    def __init__(self, size=10, resolution=.1, preset=None, a=.02, b=.2, c=-65, d=2, learning_rate=.001, tau=20,
                    assymetry=1.05, g=20, inhibitory=False, device=None):
        
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
        
        self.I = torch.zeros(size)
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.impulse = torch.zeros(size)
        self.fire = self.impulse

        self.v = torch.ones(size) * self.c
        self.u = torch.ones(size) * b*self.v

        self.resolution = resolution
        self.learning_rate = learning_rate
        self.tau = tau / resolution
        self.assymetry = assymetry
        self.g = g
        self.size = size
        self.device = device

        self.time = torch.zeros(size)
        self.spike_dt = torch.zeros(size)
        self.spike_t = torch.zeros(size)

        if device:
            self.I.to(device)
            self.impulse.to(device)
            self.v.to(device)
            self.u.to(device)




    def dynamics(self):
        self.v += self.resolution*(0.04*self.v**2 + 5*self.v + 140 - self.u + self.I)
        self.u += self.resolution*(self.a*(self.b * self.v - self.u))
        self.time += self.resolution
        self.spike_trace()


    def spike_trace(self):
        self.impulse -= self.impulse/self.tau


    def recovery(self):
        self.fire = self.impulse
        for i in range(len(self.I)):
            if self.v[i] >= 30:
                self.fire[i] += 1
                self.v[i] = self.c[i]
                self.u[i] += self.d
                self.spike_t[i] = self.time[i]
                self.spike_dt[i] = 0
            else:
                self.fire[i] = 0
        return self.fire


    def apply_current(self, signal):
        self.I = signal


    def behave(self):
        self.dynamics()
        impulse = self.recovery()
        return impulse, self.v


    def __len__(self):
        return self.size