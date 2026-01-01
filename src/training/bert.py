import torch
import numpy as np
import sklearn
import transformers
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from src.data.load_data import load_train
from src.utils.kappa import quadratic_weighted_kappa
from src.utils.plots import plot_training_history
from src.config.config import config, TrainingConfig
from src.utils.seed import set_seed
from src.data.preprocess import tokenize_dataset
from datasets import Dataset
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich import box
from rich.markdown import Markdown
import time
from rich.traceback import install


console = Console()

def print_banner(exp_name: str, config: TrainingConfig):
    """🚀 Cool animated startup banner"""
    """📱 Compact single-line banner (alternative)"""
    console.rule(title=f"🤖 BERT Essay Scoring [bold yellow]{exp_name}[/bold yellow]", style="bold cyan")
    
    summary = Table.grid(padding=(0, 1), expand=True)
    summary.add_row(f"[bold]Model:[/bold] {config.model_name}")
    summary.add_row(f"[bold]Output Dir:[/bold] {config.train_batch_size}")
    
    summary.add_row(f"[bold]Epochs:[/bold] {config.num_epoch}")
    summary.add_row(f"[bold]LR:[/bold] {config.learning_rate:.1e}")
    
    summary.add_row(f"[bold]Max length:[/bold] {config.max_length}")  
    summary.add_row(f"[bold]Weight Decay:[/bold] {config.weight_decay}")
    
    summary.add_row(f"[bold]Batch:[/bold] {config.train_batch_size}")
    summary.add_row(f"[bold]Eval Batch:[/bold] {config.eval_batch_size}")
    
    summary.add_row(f"[bold]Gradient Accumalation steps:[/bold] {config.gradient_accumalation_steps}")
    summary.add_row(f"[bold]Max grad norm:[/bold] {config.max_grad_norm}")
    summary.add_row(f"[bold]Early stopping patience:[/bold] {config.early_stopping_patience}")
    summary.add_row(f"[bold]FP16:[/bold] {config.fp16}")

    console.print(Panel(summary, title="⚙️ Config", border_style="green"))
    

def sanity_check():
    """📊 Enhanced library versions table"""
    table = Table(title="🔍 Environment Check", box=box.DOUBLE_EDGE, show_header=True, header_style="bold green")
    table.add_column("Library", style="bold blue", width=20)
    table.add_column("Version", style="green")
    table.add_row("PyTorch", torch.__version__)
    table.add_row("Transformers", transformers.__version__)
    table.add_row("NumPy", np.__version__)
    table.add_row("Scikit-learn", sklearn.__version__)
    table.add_row("Rich", "🎨 Active!")  # Rich doesn't have __version__ easily
    console.print(table)
    
    # GPU status
    if torch.cuda.is_available():
        gpu_info = f"🚀 GPU: [bold green]{torch.cuda.get_device_name(0)}[/bold green] | VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB"
        console.print(Panel(gpu_info, style="bold green"))

def build_trainer(model, tokenizer, train_ds, val_ds, TrainingConfig, compute_metric):
  
  args = TrainingArguments(
      output_dir=f"{TrainingConfig.output_dir}/{TrainingConfig.exp_name}",
      eval_strategy="epoch",
      save_strategy="epoch" , 
      
      learning_rate=TrainingConfig.learning_rate,
      per_device_train_batch_size=TrainingConfig.train_batch_size,
      per_device_eval_batch_size=TrainingConfig.eval_batch_size,
      
      gradient_accumulation_steps=TrainingConfig.gradient_accumalation_steps,
      num_train_epochs=TrainingConfig.num_epoch,
      
      weight_decay=TrainingConfig.weight_decay,
      max_grad_norm=TrainingConfig.max_grad_norm,
      
      load_best_model_at_end=True,
      metric_for_best_model="qwk",
      greater_is_better=True,
      save_total_limit=1,  # ← KEEP ONLY BEST
      save_safetensors=True,
      fp16=TrainingConfig.fp16,
      logging_steps=50,
      report_to="none" 
  )
  
  trainer = Trainer(
      model = model,
      args = args,
      train_dataset=train_ds,
      eval_dataset=val_ds,
      processing_class=tokenizer,
      data_collator=DataCollatorWithPadding(tokenizer),
      compute_metrics=compute_metric,
       callbacks=[EarlyStoppingCallback(early_stopping_patience=TrainingConfig.early_stopping_patience)],
      
    )
  return trainer


def compute_metrics(eval_pred):
   predict, label = eval_pred
   predict = predict.squeeze()
   predict = np.clip(predict,1, 6)
   qwk = quadratic_weighted_kappa(label, predict)
   return {"qwk": qwk}


class BertForEssayScoring(torch.nn.Module):
  def __init__(self, model_name:str, dropout:float):
    super().__init__()
    self.bert = AutoModel.from_pretrained(model_name)
    self.dropout = torch.nn.Dropout(dropout)
    self.regressor = torch.nn.Linear(self.bert.config.hidden_size, 1)
    

  def forward(self, input_ids, attention_mask, labels=None):
      outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
  
      pooled = outputs.last_hidden_state.mean(dim=1)
      pooled = self.dropout(pooled)
      logits = self.regressor(pooled).squeeze(-1)
      
      loss = None
      if labels is not None:
        loss = torch.nn.functional.mse_loss(logits, labels.float())
      return {"loss": loss, "logits": logits}


if __name__ == "__main__":
  train_config = TrainingConfig()
  set_seed(train_config.random_state)
  console.print(f"[bold green]Setting seed and starting experimentation with following config")
  
  print_banner(train_config.exp_name, train_config)
  sanity_check()
  
  df = load_train()
  df["score"] = df["score"].astype("float32")
  
  train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["score"], random_state=train_config.random_state)

  train_ds = Dataset.from_pandas(train_df.reset_index(drop=True))
  val_ds = Dataset.from_pandas(val_df.reset_index(drop=True))

  tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
  
  train_ds = tokenize_dataset(train_ds, tokenizer, train_config.max_length)
  val_ds = tokenize_dataset(val_ds, tokenizer, train_config.max_length)

  train_ds = train_ds.rename_column("score", "labels")
  val_ds = val_ds.rename_column("score", "labels")
  
  
  
  train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
  val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
  
  model = BertForEssayScoring(model_name=train_config.model_name, dropout=train_config.dropout)
 
  trainer = build_trainer(model, tokenizer, train_ds, val_ds, train_config, compute_metrics)
  
  console.print(f"Device: {trainer.args.device}")
  if torch.cuda.is_available():
    console.print(f"GPU Name: {torch.cuda.get_device_name(0)}")
  
  console.print(Panel.fit("🚀 Starting Training! Press Ctrl+C to stop", style="bold green"))
  
  trainer.train()
  

  final_metrics = trainer.evaluate()
  console.print(f"\n🏆 [bold green]Final QWK: {final_metrics['eval_qwk']:.4f}[/bold green]")
  

  exp_dir = f"{train_config.output_dir}/{train_config.exp_name}"
  trainer.save_model(exp_dir)
  tokenizer.save_pretrained(exp_dir)
  console.print(f"💾 Saved to [bold blue]{exp_dir}[/bold blue]")
  

  history = trainer.state.log_history
  plot_training_history(history,exp_dir)
  console.print("\n🎉 [bold green]Training Complete![/bold green] ✨")

