from torch.utils.data import Dataset, DataLoader

class Data(Dataset):
    def __init__(self, transform=None):
        super(Data, self).__init__()
        self.transform = transform
        self.x = None
        self.y = None

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]

        if self.transform:
            x = self.transform(x)
            y = self.transform(y)

        return x, y

if __name__ == '__main__':
    trainset = Data()
    it = iter()
    print(next(it))

