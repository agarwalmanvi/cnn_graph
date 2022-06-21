import torch
from torch.utils.data import Dataset
import os
import wfdb
import numpy as np


class Apnea(Dataset):
    def __init__(
            self,
            path: str,
            ds_type: str = "train",
    ):
        super(Apnea, self).__init__()
        self.path = path
        self.ds_type = ds_type

        if ds_type == "train":
            n_ = ["0" + str(i) for i in range(1, 10)] + [str(i) for i in list(range(10, 21))]
            # fnames = ["a" + i for i in n_] + ["b" + i for i in n_[:5]] + ["c" + i for i in n_[:10]]
            fnames = ["a" + i for i in n_[:5]]
        elif ds_type == "test":
            # the testing files don't come with labels :/
            # n_ = ["0" + str(i) for i in range(1, 10)] + [str(i) for i in list(range(10, 36))]
            # fnames = ["x" + i for i in n_[:3]]
            n_ = ["0" + str(i) for i in range(1, 10)] + [str(i) for i in list(range(10, 21))]
            fnames = ["a" + i for i in n_[5:7]]
        else:
            raise Exception("dataset type cannot be identified.")

        # # # Get data and labels # # #

        samples = None
        labels = []

        for fname in fnames:

            print("Processing ", fname, "...")

            # import all relevant data
            signal = wfdb.rdrecord(os.path.join(self.path, fname)).__dict__['p_signal']
            annotation = wfdb.rdann(os.path.join(self.path, fname), 'apn')
            sample_boundaries = annotation.__dict__["sample"]
            sample_labels = annotation.__dict__["symbol"]

            for boundary_idx in range(len(sample_boundaries) - 1):
                # select the one minute segment
                sample_ = signal[sample_boundaries[boundary_idx]:sample_boundaries[boundary_idx + 1]]
                samples = sample_ if samples is None else np.concatenate((samples, sample_), 1)

                # select corresponding label
                label_ = sample_labels[boundary_idx]
                assert (label_ == "A") or (label_ == "N")
                label_ = 0 if label_ == "N" else 1
                labels.append(label_)

        # cast samples to torch tensor and transpose to shape: nb_samples x seq_len
        self.X = torch.transpose(torch.tensor(samples, dtype=torch.float), 0, 1)
        # labels: nb_samples x 1
        self.y = torch.unsqueeze(torch.tensor(labels, dtype=torch.float), -1)

        # normalize data
        self.X = (self.X - torch.min(self.X)) / (torch.max(self.X) - torch.min(self.X))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx, :]
