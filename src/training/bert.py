import torch
import numpy as np
import sklearn
import transformers
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, DataCollatorWithPadding, EarlyStoppingCallback
from src.data.load_data import load_train
from src.utils.kappa import quadratic_weighted_kappa
from src.utils.plots import plot_training_history
from src.config.config import TrainingConfig
from src.utils.seed import set_seed
from src.data.preprocess import tokenize_dataset
from src.training.pooling import Pooling
from datasets import Dataset
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
console = Console()

def to_ordinal_labels(score, num_classes=6):
    return [int(score > i) for i in range(1, num_classes)]

def add_ordinal_labels(example):
    example["labels"] = to_ordinal_labels(example["labels"])
    return example
def print_banner(exp_name: str, config: TrainingConfig):
    """🚀 Cool animated startup banner"""
    """📱 Compact single-line banner (alternative)"""
    console.rule(title=f"🤖 BERT Essay Scoring [bold yellow]{exp_name}[/bold yellow]", style="bold cyan")
    
    summary = Table.grid(padding=(0, 1), expand=True)
    summary.add_row(f"[bold]Model:[/bold] {config.model_name}")
    summary.add_row(f"[bold]Output Dir:[/bold] {config.output_dir}/{config.exp_name}")
    
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
      report_to="none",
      remove_unused_columns=False
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
   logits, labels = eval_pred
   probs = 1 / (1 + np.exp(-logits))
   preds = (probs > 0.5).sum(axis=1) + 1
   true = labels.sum(axis=1) + 1
   qwk = quadratic_weighted_kappa(true, preds)
   return {"qwk": qwk}


class BertForEssayScoring(torch.nn.Module):
  def __init__(self, model_name:str, dropout:float, pooling:str, num_clasess=6, num_prompts = 8):
    super().__init__()
    self.encoder = AutoModel.from_pretrained(model_name)    
    hidden_size = self.encoder.config.hidden_size
    
    self.num_classes = num_clasess
    self.pooler = Pooling(hidden_size, pooling)
   
    
    self.shared = torch.nn.Sequential(
        torch.nn.Linear(hidden_size, hidden_size*2), 
        torch.nn.GLU(),
        torch.nn.Dropout(dropout)
      )
    
    
    self.prompt_heads = torch.nn.ModuleDict({
      str(i):torch.nn.Linear(hidden_size,num_clasess-1) for i in range(1, num_prompts+1)
    })
    
  def forward(self, input_ids, attention_mask, prompt_id, labels=None):
      
      outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
  
      pooled =  self.pooler(outputs.last_hidden_state, attention_mask)
      shared = self.shared(pooled)
      logits = torch.zeros(shared.size(0), self.num_classes-1, device=shared.device,  dtype=shared.dtype)
      
      for pid in prompt_id.unique():
          mask = prompt_id == pid
          logits[mask] = self.prompt_heads[str(pid.item())](shared[mask])
      
      loss = None
      if labels is not None:
         loss = self.ordinal_loss(logits, labels.float())
      return {"loss": loss, "logits": logits}
  
  def ordinal_loss(self, logits, labels):
      bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
      penalty = torch.relu(logits[:, :-1] - logits[:, 1:]).mean()
      return bce + 0.1 * penalty


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
  
  train_ds = train_ds.add_column("prompt_id", [1] * len(train_ds))
  val_ds   = val_ds.add_column("prompt_id", [1] * len(val_ds))
  

  train_ds = train_ds.map(add_ordinal_labels)
  val_ds   = val_ds.map(add_ordinal_labels)
  
 
  train_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels", "prompt_id"])
  val_ds.set_format("torch", columns=["input_ids", "attention_mask", "labels", "prompt_id"])
  
  model = BertForEssayScoring(model_name=train_config.model_name, dropout=train_config.dropout, pooling=train_config.pooling)
  # model.encoder.gradient_checkpointing_enable()
  # model.encoder.config.use_cache = False
  
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

