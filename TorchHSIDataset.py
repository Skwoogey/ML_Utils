import torch as t
import numpy as np

class HSIDataset(t.utils.data.Dataset):
    def __init__(self, samples, labels):
        super(HSIDataset, self).__init__()

        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        #print(self.samples[idx].shape)
        sample = np.ascontiguousarray(self.samples[idx])
        label = self.labels[idx]

        return sample, label
        
