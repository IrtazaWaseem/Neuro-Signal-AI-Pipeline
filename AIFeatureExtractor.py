import numpy as np
class AIFeatureExtractor:
    def __init__(self,clean_signal):
        self.clean_signal = clean_signal
    def segment_into_epochs(self,epoch_size):
        valid_length = (len(self.clean_signal) // epoch_size) * epoch_size
        trimmed_signal = self.clean_signal[:valid_length]
        self.seg_epochs = trimmed_signal.reshape(-1, epoch_size)
    def build_training_matrix(self):
        maximum = np.max(self.seg_epochs,axis=1)
        minimum = np.min(self.seg_epochs, axis=1)
        sum_signal = np.sum(self.seg_epochs, axis=1)
        features = np.column_stack((maximum,minimum,sum_signal))
        self.master_ml_matrix = np.hstack((self.seg_epochs,features))
    def split_train_test(self):
        split_index = int(0.8 * len(self.master_ml_matrix))
        training_data,testing_data = np.vsplit(self.master_ml_matrix,[split_index])
        return training_data,testing_data




