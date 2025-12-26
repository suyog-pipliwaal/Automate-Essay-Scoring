import pandas as pd
from pathlib import Path
from src.config.config import config
DATA_DIR = Path(config.data_dir)

def load_train():
	return pd.read_csv(DATA_DIR / "train.csv")

def load_test():
	return pd.read_csv(DATA_DIR / "test.csv")

if __name__ == "__main__":
    print("Train:", load_train().shape)
    print("Test:", load_test().shape)

