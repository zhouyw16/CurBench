import numpy as np
from torch_geometric.datasets import TUDataset


def get_tudataset_dataset(data_name, data_dir='data/tudataset', split=[8, 1, 1]):
    dataset = TUDataset(root=data_dir, name=data_name)
    dataset.__setattr__('name', data_name)

    def split_dataset(dataset, split):
        split = np.array(split)
        assert (split > 0).all(), 'Assert Error: split ratio should be positive'

        split = split.cumsum() / split.sum()
        idx_valid = int(split[0] * len(dataset))
        idx_test = int(split[1] * len(dataset))

        dataset = dataset.shuffle()
        return dataset[:idx_valid], dataset[idx_valid:idx_test], dataset[idx_test:]
    
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, split)
    return dataset, train_dataset, valid_dataset, test_dataset
