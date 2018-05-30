from .scut5500_dataset import Scut5500Dataset


class DatasetFactory:
    @classmethod
    def create_dataset(
            cls, name, data_dir, data_list, input_size, transform_method
        ):
        if name == 'SCUT5500':
            dataset = Scut5500Dataset(
                data_dir, data_list, input_size, transform_method
            )
        else:
            raise KeyError('Unrecognized dataset: ' + name)
        return dataset
