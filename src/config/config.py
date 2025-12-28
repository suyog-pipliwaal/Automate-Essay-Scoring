from pydantic import BaseModel

class SystemConfig(BaseModel):
  data_dir:str      = "data/raw"
  raw_train_csv:str = "data/raw/train.csv"
  random_state:int  = 42
  model_name:str    = "bert-base-uncased"


config = SystemConfig()