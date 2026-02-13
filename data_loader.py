from src.data.data_loader import GoDataset


if __name__ == "__main__":
    for split in ("train", "val", "test"):
        ds = GoDataset(data_dir=f"dataset/{split}")
        print(f"{split}: {len(ds)} samples: first item: {ds[0]}")
