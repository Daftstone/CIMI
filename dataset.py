from torch.utils.data import Dataset

class My_dataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        fn = self.data[index]
        label = self.labels[index]
        return fn, label

    def __len__(self):
        return len(self.data)