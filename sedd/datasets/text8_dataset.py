from torch.utils.data import Dataset, DataLoader
import torch

import os
from dataclasses import dataclass
from typing import Dict, Tuple, List

def ensure_text8(text8_path: str):
    """
    Ensures text8 exists at `text8_path`.
    If not, downloads and extracts it.
    """
    if os.path.exists(text8_path):
        return

    os.makedirs(os.path.dirname(text8_path) or ".", exist_ok=True)
    zip_path = text8_path + ".zip"

    print("text8 not found. Downloading...")
    urllib.request.urlretrieve("http://mattmahoney.net/dc/text8.zip", zip_path)

    print("Extracting...")
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(os.path.dirname(text8_path) or ".")

    os.remove(zip_path)
    print("text8 ready.")

@dataclass
class Text8Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def size(self) -> int:
        return len(self.itos)

    def encode(self, s: str) -> torch.Tensor:
        return torch.tensor([self.stoi[c] for c in s], dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return "".join(self.itos[i] for i in ids)

    def batch_decode(self, ids: Union[torch.Tensor, List[List[int]]]):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return ["".join(self.itos[i] for i in seq) for seq in ids]


class Text8Dataset(Dataset):
    def __init__(
        self,
        text8_path: str,
        block_size: int = 256,
        split: str = 'train',
        train_frac: float = 0.9,
        vocab: Text8Vocab = None,
        num_examples: int = -1
    ):
        assert split in {"train", "val"}
        assert 0.0 < train_frac < 1.0
        assert block_size >= 2

        ensure_text8(text8_path)
        with open(text8_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        # Build vocab if not provided
        if vocab is None and split == "train":
            chars = sorted(set(text))
            stoi = {ch: i for i, ch in enumerate(chars)}
            itos = chars
            vocab = Text8Vocab(stoi=stoi, itos=itos)
        # self.vocab = vocab

        data = self.vocab.encode(text)  # [N]
        n = len(data)
        split_idx = int(n * train_frac)

        if split == "train":
            self.data = data[:split_idx]
        else:
            self.data = data[split_idx:]

        if num_examples > 0:
            print(f"Subsampling dataset to {num_examples} examples")
            self.data = self.data.select(range(num_examples))
        self.block_size = block_size

        self.num_examples = min(num_examples, len(self.data)-self.block_size+1)

    def __len__(self) -> int:
        return self.num_examples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx : idx + self.block_size]


def make_text8_loaders(
    data_path: str,
    block_size: int = 64,
    batch_size: int = 256,
    num_workers: int = 1,
    train: bool = True,
    pin_memory: bool = True
):
    # Build train first to create vocab, then reuse for val
    # train_ds = Text8Dataset(data_path, block_size=block_size, split="train")
    # val_ds = Text8Dataset(data_path, block_size=block_size, split="val", vocab=train_ds.vocab)
    if train:
        dataset = Text8Dataset(data_path, block_size=block_size, split='train')
    else:
        dataset = Text8Dataset(data_path, block_size=block_size, split='val')

    dataloader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     drop_last=False,
    # )
    # return train_loader, val_loader, train_ds.vocab
    return dataloader, dataset.vocab


# ---- Example usage ----
if __name__ == "__main__":
    text8_path = "text8"  # path to the extracted text8 file
    train_loader, val_loader, vocab = make_text8_loaders(
        text8_path=text8_path,
        block_size=256,
        batch_size=64,
    )

    xb, yb = next(iter(train_loader))
    print("x:", xb.shape, "y:", yb.shape, "vocab:", vocab.size)
    print("sample:", vocab.decode(xb[0][:80]))
