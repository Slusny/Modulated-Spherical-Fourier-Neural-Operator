

import os
import pickle
import zipfile
from typing import Any


class FakeStorage:
    def __init__(self):
        import torch

        self.dtype = torch.float32
        self._untyped_storage = torch.UntypedStorage(0)


class UnpicklerWrapper(pickle.Unpickler):
    def __init__(self, file, **kwargs):
        super().__init__(file, **kwargs)

    def persistent_load(self, pid: Any) -> Any:
        return FakeStorage()


def tidy(x):
    if isinstance(x, dict):
        return {k: tidy(v) for k, v in x.items()}

    if isinstance(x, list):
        return [tidy(v) for v in x]

    if isinstance(x, tuple):
        return tuple([tidy(v) for v in x])

    if x is None:
        return None

    if isinstance(x, (int, float, str, bool)):
        return x

    return str(type(x))


def peek(path):
    with zipfile.ZipFile(path, "r") as f:
        data_pkl = None
        for b in f.namelist():
            if os.path.basename(b) == "data.pkl":
                if data_pkl is not None:
                    raise Exception(
                        f"Found two data.pkl files in {path}: {data_pkl} and {b}"
                    )
                data_pkl = b

        unpickler = UnpicklerWrapper(f.open(data_pkl, "r"))
        x = tidy(unpickler.load())
        return tidy(x)
