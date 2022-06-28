import numpy as np
import wfdb
import os


def get_apnea_data(fnames, path):
    """
    Used to import data from apnea dataset and preprocess into one minute segments
    :param fnames: all filenames that you want to import
    :param path: where the apnea data is located
    :return: tuple of numpy arrays containing samples and labels
    """
    samples = None
    labels = []

    for fname in fnames:

        print("Processing ", fname, "...")

        # import all relevant data
        signal = wfdb.rdrecord(os.path.join(path, fname)).__dict__['p_signal']
        annotation = wfdb.rdann(os.path.join(path, fname), 'apn')
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

    labels = np.array(labels)
    return (samples, labels)
