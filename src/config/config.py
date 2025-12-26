from pydantic import BaseModel

class SystemConfig(BaseModel):
  data_dir:str      = "data/raw"
  raw_train_csv:str = "data/raw/train.csv"
  


config = SystemConfig()