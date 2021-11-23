import os
import logging
import pandas as pd


def open_csv(filename, format=None):
    return pd.read_csv(filename, sep=';', low_memory=False)


class Model:
    def __init__(self, data_folder=None, train_prefix='_train', test_prefix='_test'):
        if data_folder is None:
            data_folder = os.path.abspath(os.path.join('.', 'data'))

        self.data_folder = data_folder
        self.train_prefix = train_prefix
        self.test_prefix = test_prefix

        self.datasets = {'test': {}, 'train': {}}

    def load(self):
        for entry in os.scandir(self.data_folder):
            entry_name = entry.name.split('.')[0]
            entry_path = entry.path

            if entry_path.endswith('.csv') and entry.is_file():
                if entry_name.find(self.train_prefix) > -1:
                    logging.debug(f'found train file {entry_path}')
                    self.datasets['train'][entry_name] = self.load_entry(
                        entry_name, entry_path)
                    continue

                if entry_name.find(self.test_prefix) > -1:
                    logging.debug(f'found test file {entry_path}')
                    self.datasets['test'][entry_name] = self.load_entry(
                        entry_name, entry_path)
                    continue

                logging.debug(f'found data file {entry_path}')
                self.datasets[entry_name] = self.load_entry(
                    entry_name, entry_path)

        return self

    def load_entry(self, entry_name, entry_path):
        df = open_csv(entry_path)
        df.columns = [
            f'{entry_name}_{"_".join(col_name.split())}' if 'id' not in col_name else col_name for col_name in df.columns]
        return df


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG,
                        format='%(levelname)s - %(message)s')

    model = Model()
    model.load()
    print(model.datasets.keys())
