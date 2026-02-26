# finance_dataloaders.py
from typing import Optional, Dict, Any
from torch.utils.data import DataLoader

from Datasets.multi_asset_dataset import Dataset_Finance_MultiAsset


class DataLoaders:
    """
    Convenience wrapper that instantiates Dataset_Finance_MultiAsset for
    train/val/test and exposes ready-to-use DataLoaders.
    """
    def __init__(
        self,
        dataset_kwargs: Dict[str, Any],
        datasetCLS=Dataset_Finance_MultiAsset,
        batch_size_train: int = 64,
        batch_size_eval: int = 256,
        num_workers: int = 0,
        pin_memory: bool = False,
        drop_last_train: bool = False,
        drop_last_eval: bool = False,
        persistent_workers: bool = False,  # set True if num_workers > 0 and long-lived
        prefetch_factor: Optional[int] = None  # only used if num_workers > 0
        ):
        """
        Args:
            dataset_kwargs: kwargs passed to Dataset_Finance_MultiAsset except 'split'.
                            e.g. {
                                "root_path": "...",
                                "data_path": "polygon/data_raw_1m",
                                "size": [512, 96],  # label_len ignored by your __getitem__
                                "rolling_window": 256,
                                "use_time_features": True,
                                "time_col_name": "timestamp",
                                "train_split": 0.7,
                                "test_split": 0.15,
                                "tickers": ["AAPL", "AMZN"],
                            }
            batch_size_train: int = 64,
            batch_size_eval: int = 256,
            num_workers: int = 0,
            pin_memory: bool = False,
            drop_last_train: bool = False,
            drop_last_eval: bool = False,
            persistent_workers: bool = False,  # set True if num_workers > 0 and long-lived
            prefetch_factor: Optional[int] = None  # only used if num_workers > 0
        """

        self.datasetCLS = datasetCLS

        self.batch_size_train = batch_size_train
        self.batch_size_eval = batch_size_eval

        if "split" in dataset_kwargs.keys():
            del dataset_kwargs["split"]

        if "batch_size" in dataset_kwargs.keys():
            del dataset_kwargs["batch_size"]

        self.dataset_kwargs = dataset_kwargs
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last_train = drop_last_train
        self.drop_last_eval = drop_last_eval
        self.persistent_workers = persistent_workers
        self.prefetch_factor = prefetch_factor

        self.train = self.train_dataloader()
        self.valid = self.val_dataloader()
        self.test = self.test_dataloader()

    def train_dataloader(self):
        return self._make_loader("train", shuffle=True)

    def val_dataloader(self):
        return self._make_loader("val", shuffle=False)

    def test_dataloader(self):
        return self._make_loader("test", shuffle=False)

    def _make_loader(self, split, shuffle=False):
        dataset = self.datasetCLS(split=split, **self.dataset_kwargs)
        effective_prefetch_factor = self.prefetch_factor if self.num_workers > 0 else None
        return DataLoader(
            dataset,
            batch_size=self.batch_size_train if split == "train" else self.batch_size_eval,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last_train if split == "train" else self.drop_last_eval,
            persistent_workers=self.persistent_workers,
            prefetch_factor=effective_prefetch_factor,
        )
        

    # Optional: aliases for common patterns
    def train_loader(self) -> DataLoader:
        return self.train

    def val_loader(self) -> DataLoader:
        return self.valid

    def test_loader(self) -> DataLoader:
        return self.test
