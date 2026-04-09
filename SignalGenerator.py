import numpy as np
class SignalGenerator:
    def __init__(self,total_points):
        self.time_axis = np.linspace(0,10*np.pi,total_points)
    def create_brainwave(self,frequency,amplitude):
        result = np.sin(self.time_axis*frequency)
        self.clean_signal = amplitude*result
        return self.clean_signal
    def inject_static(self,noise_intensity):
        noise = np.random.randn(len(self.clean_signal)) * noise_intensity
        self.noisy_signal = noise + self.clean_signal
        return self.noisy_signal
