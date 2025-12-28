import torch
import numpy as np
import sklearn
import transformers
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from src.data.load_data import load_train, load_test
from src.utils.kappa import quadratic_weighted_kappa
from src.utils.timeit import timeit
from src.config.config import config
from rich.console import Console
from rich.table import Table
from datasets import Dataset
console = Console()
EXP_NAME = "exp_2_bert_finetune"
ROOT_DIR = f"src/models/{EXP_NAME}"
RANDOM_STATE = 42
def sanity_check() -> None:
  table = Table(show_header=True, header_style="bold magenta")
  table.add_column("Lib Name", style="dim", width=25)
  table.add_column("Version")
  table.add_row("Pytorch", torch.__version__)
  table.add_row("Huggingface Transformers", transformers.__version__)
  table.add_row("Numpy", np.__version__)
  table.add_row("sklean", sklearn.__version__)
  console.print(table)


def compute_metrics(eval_pred):
   predict, label = eval_pred
   predict = np.clip(np.round(predict),1, 6)
   qwk = quadratic_weighted_kappa(label, predict)
   return {"qwk": qwk}


class BertForEassyScoring(torch.nn.Module):
  def __init__(self, model_name="bert-base-uncased"):
    super().__init__()
    self.bert = AutoModel.from_pretrained(model_name)
    self.dropout = torch.nn.Dropout(0.2)
    self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)
  def forward(self, input_ids, attention_mask, labels=None):
      outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      pooled = outputs.last_hidden_state[:, 0]
      pooled = self.dropout(pooled)
      logits = self.regressor(pooled).squeeze(-1)
      
      loss = None
      if labels is not None:
        loss_fun = torch.nn.MSELoss()
        loss = loss_fun(logits, labels.float())
      return {"loss": loss, "logits": logits}
        
def save_model():
  pass



if __name__ == "__main__":
  sanity_check()
  df = load_train()
  model_name = config.model_name
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  
  train_df, val_df = train_test_split(df, test_size=0.2, random_state=config.random_state)
  
  train_df = train_df.reset_index(drop=True)
  val_df = val_df.reset_index(drop=True)
  
  train_ds = Dataset.from_pandas(train_df)
  val_ds = Dataset.from_pandas(val_df)
  
  def tokenize_fn(batch):
    return tokenizer(
        batch["full_text"],
        truncation=True,
        padding="max_length",
        max_length=512
    )
  
  train_ds = train_ds.map(tokenize_fn, batched=True)
  val_ds = val_ds.map(tokenize_fn, batched=True)
  
  train_ds = train_ds.rename_column("score", "labels")
  val_ds = val_ds.rename_column("score", "labels")
  
  
  train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
  val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
  
  training_args = TrainingArguments(
    output_dir=ROOT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    weight_decay=0.01,
    
    load_best_model_at_end=True,
    metric_for_best_model="qwk",
    greater_is_better=True,
    save_total_limit=1
    
    logging_steps=50,
    fp16=torch.cuda.is_available()
  ) 
  model = BertForEassyScoring()
  if torch.cuda.is_available():
    console.print(f"GPU Name: {torch.cuda.get_device_name(0)}")
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
  )
  console.print(f"[bold green]Training will run on device:[/bold green] {training_args.device}")
  trainer.train()
  trainer.save_model(ROOT_DIR)
  tokenizer.save_pretrained(ROOT_DIR)
  
  console.print(f"[bold cyan]Best model saved to:[/bold cyan] {ROOT_DIR}")


