import torch
from torch.utils.data import Dataset
from lib.mlggm.preprocess_data import get_apnea_data
import numpy as np
import os


class Apnea(Dataset):
    def __init__(
            self,
            path: str,
            ds_type: str
    ):
        super(Apnea, self).__init__()
        self.path = path

        n_ = ["0" + str(i) for i in range(1, 10)] + [str(i) for i in list(range(10, 21))]
        fnames = ["a" + i for i in n_] + ["b" + i for i in n_[:5]] + ["c" + i for i in n_[:10]]

        data = np.load(os.path.join(path, ds_type+"_"+"apnea_data.npz"))
        samples, labels = data['samples'], data['labels']

        # try:
        #     data = np.load(os.path.join(path, "apnea_data.npz"))
        #     samples, labels = data['samples'], data['labels']
        #     print("Loaded data from cache...")
        # except:
        #     samples, labels = get_apnea_data(fnames=fnames, path=path)
        #     print("Loaded data from original source, saving for later use...")
        #     with open(os.path.join(path, "apnea_data.npz"), "wb") as f:
        #         np.savez(f, samples=samples, labels=labels)

        # cast samples to torch tensor and transpose to shape: nb_samples x seq_len
        self.X = torch.transpose(torch.tensor(samples, dtype=torch.float), 0, 1)
        # labels: nb_samples x 1
        self.y = torch.unsqueeze(torch.tensor(labels, dtype=torch.float), -1)

        # normalize data
#         self.X = (self.X - torch.min(self.X)) / (torch.max(self.X) - torch.min(self.X))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx, :]
