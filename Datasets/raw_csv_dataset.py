"""Minimal PyTorch dataset for numeric rows in a CSV file."""

from __future__ import annotations

import os
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd

__all__ = ["RawCSVDataset", "create_raw_csv_dataset"]


class RawCSVDataset(Dataset):
    """Load a CSV file into memory and expose its numeric rows as torch tensors."""

    def __init__(
        self,
        csv_path: str,
        columns: Optional[Sequence[str]] = None,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        transform: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        dropna: bool = False,
        na_values: Optional[Iterable[str]] = None,
    ) -> None:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        self._csv_path = csv_path
        self._transform = transform

        frame = pd.read_csv(
            csv_path,
            usecols=columns,
            na_values=na_values,
        )

        if dropna:
            frame = frame.dropna()

        if columns is None:
            # Retain only numeric columns when auto-selecting features.
            numeric_frame = frame.select_dtypes(include=[np.number])
            if numeric_frame.empty:
                raise ValueError(
                    "No numeric columns detected. Specify 'columns' to select features explicitly."
                )
            frame = numeric_frame

        values = frame.to_numpy(dtype=np.float64, copy=False)
        if np.isnan(values).any():
            raise ValueError("Detected NaN values in the CSV data. Clean the file or set dropna=True.")

        tensor = torch.as_tensor(values, dtype=dtype)
        if device is not None:
            tensor = tensor.to(device)

        self._data = tensor

    def __len__(self) -> int:
        return self._data.shape[0]

    def __getitem__(self, index: int) -> torch.Tensor:
        sample = self._data[index]
        if self._transform is not None:
            sample = self._transform(sample)
        return sample

    @property
    def feature_shape(self) -> torch.Size:
        """Return the trailing shape of an item."""
        return self._data.shape[1:]

    @property
    def csv_path(self) -> str:
        """Return the source CSV path."""
        return self._csv_path


def create_raw_csv_dataset(
    csv_path: str,
    **kwargs,
) -> RawCSVDataset:
    """Factory helper returning a RawCSVDataset for convenience."""

    return RawCSVDataset(csv_path=csv_path, **kwargs)
