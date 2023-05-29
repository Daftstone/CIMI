from torch.utils.data import Dataset

class My_dataset(Dataset):
<<<<<<< HEAD
    def __init__(self, data, evidences, labels):
        self.data = data
        self.labels = labels
        self.evidences = evidences
=======
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
>>>>>>> a507ac7 (init)

    def __getitem__(self, index):
        fn = self.data[index]
        label = self.labels[index]
<<<<<<< HEAD
        ev = self.evidences[index]
        return fn, ev, label
=======
        return fn, label
>>>>>>> a507ac7 (init)

    def __len__(self):
        return len(self.data)