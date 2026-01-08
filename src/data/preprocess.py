from transformers import AutoTokenizer

def tokenize_dataset(dataset, tokenizer, max_length):
  def tokenize(batch):
    return tokenizer(batch['full_text'], truncation=True, max_length=max_length,  return_token_type_ids=False)
  return dataset.map(tokenize, batched=True)