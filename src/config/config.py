from pydantic import BaseModel
from dataclasses import dataclass
class SystemConfig(BaseModel):
  data_dir:str      = "data/raw"
  raw_train_csv:str = "data/raw/train.csv"
  random_state:int  = 42
  model_name:str    = "bert-base-uncased"

@dataclass
class TrainingConfig:
  exp_name:str = "exp_bert_more_epoch"
  output_dir:str = "src/models"
  
  model_name:str = "bert-base-uncased"
  max_length:int = 512
  dropout:float = 0.2
  

  test_size:float = 0.2
  random_state:int = 42
  
  learning_rate:float = 2e-5
  weight_decay:float = 0.01
  num_epoch:int = 9
  
  train_batch_size:int = 8
  eval_batch_size: int = 8
  
  gradient_accumalation_steps:int = 2
  max_grad_norm:float = 1.0
  
  early_stopping_patience:int = 2
  
  fp16:bool = True
  
config = SystemConfig()