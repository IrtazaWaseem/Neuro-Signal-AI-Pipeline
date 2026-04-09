import numpy as np
class SignalFilter:
    def __init__(self,noisy_signal):
        self.noisy_signal = noisy_signal

    def hardware_clips(self,min_voltage,max_voltage):
        self.clipped_signal = np.clip(self.noisy_signal,min_voltage,max_voltage)
        return self.clipped_signal
    def remove_statistical_outliers(self):
        mean = np.mean(self.clipped_signal)
        std = np.std(self.clipped_signal)
        upper_bound = mean+(2*std)
        lower_bound = mean-(2*std)
        self.filtered_signal = np.where((self.clipped_signal>=lower_bound)&(self.clipped_signal<=upper_bound),self.clipped_signal,mean)
        return self.filtered_signal
