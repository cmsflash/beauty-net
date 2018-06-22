from argparse import Namespace

from torch.utils.data import DataLoader


_DATA_LOADER_CONFIGS = {
    'train': Namespace(split_name='Training', shuffle=True, drop_last=True),
    'val': Namespace(
        split_name='Validation', shuffle=False, drop_last=False
    )
}


def create_data_loader(input_config, split, pin_memory=True):
    loader_config = _DATA_LOADER_CONFIGS[split]
    dataset = input_config.dataset(input_config.config)
    print('{} size: {}'.format(loader_config.split_name, len(dataset)))
    data_loader = DataLoader(
        dataset,
        input_config.batch_size,
        shuffle=loader_config.shuffle,
        num_workers=1,
        pin_memory=pin_memory,
        drop_last=loader_config.drop_last
    )
    return data_loader
