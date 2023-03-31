import torch

from torch.utils.data import Dataset, DataLoader


class MnistDataset(Dataset): # custom dataset class

    def __init__(self, data, labels, flatten=True):
        self.data = data  # x
        self.labels = labels # y, LongTensor
        self.flatten = flatten # CNN의 경우, flattern=False

        super().__init__()

    def __len__(self):
        return self.data.size(0) # sample의 수

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.labels[idx]

        if self.flatten:
            x = x.view(-1) # flatten 수행, (28, 28) -> (784, )

        return x, y # |x| = (bs, 784) |y| = (1, )

def load_mnist(is_train=True, flatten=True):
    from torchvision import datasets, transforms

    dataset = datasets.MNIST(
        '../data', train=is_train, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])
    )

    x = dataset.data.float() / 255.
    y = dataset.targets

    if flatten:
        x = x.view(x.size(0), -1)

    return x, y


def get_loaders(config):
    x, y = load_mnist(is_train=True, flatten=False)

    train_cnt = int(x.size(0) * config.train_ratio)
    valid_cnt = x.size(0) - train_cnt

    flatten = True if config.models == 'fc' else False

    # Shuffle dataset to split into train/valid set.
    indices = torch.randperm(x.size(0)) # |x| = (60000, 28, 28)
    train_x, valid_x = torch.index_select(
        x,
        dim=0,
        index=indices,
    ).split([train_cnt, valid_cnt], dim=0) # 600000을 train과 valid로 나눔

    train_y, valid_y = torch.index_select(
        y,
        dim=0,
        index=indices,
    ).split([train_cnt, valid_cnt], dim=0)

    train_loader = DataLoader(
        dataset=MnistDataset(train_x, train_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True, # training은 무조건 shuffling, 안하면 학습이 되지 않음
    )

    valid_loader = DataLoader(
        dataset=MnistDataset(valid_x, valid_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=True,
    )

    test_x, test_y = load_mnist(is_train=False, flatten=False)
    test_loader = DataLoader(
        dataset=MnistDataset(test_x, test_y, flatten=flatten),
        batch_size=config.batch_size,
        shuffle=False, # 보통 False, 일부 수정된 부분에 대한 동작/결과를 보고자 할 때,
                       # 같은 테스트셋으로 진행해야 그 수정이 올바르게 되었는지 알 수 있음
    )

    return train_loader, valid_loader, test_loader


