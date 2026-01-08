# Automated Essay Scoring: A Deep Learning Approach

**GitHub Repository:** [https://github.com/suyog-pipliwaal/Automate-Essay-Scoring](https://github.com/suyog-pipliwaal/Automate-Essay-Scoring)

---

## Section 1: Context (Brief)

This project originated from the **Learning Agency Lab – Automated Essay Scoring 2.0** Kaggle competition, where the objective was to automatically predict essay scores (1-6 scale) that match human rater evaluations. The primary goal was to enhance deep learning skills and gain hands-on experience with state-of-the-art natural language processing technologies. The project systematically explored the evolution from classical NLP methods (TF-IDF + traditional ML) to advanced transformer-based architectures (BERT [Devlin et al., 2018], DeBERTa [He et al., 2021]), ultimately achieving significant performance improvements from a baseline QWK of 0.7044 to over 0.80 through careful architectural design, custom ordinal regression loss functions [Frank & Hall, 2001], and prompt-specific modeling approaches.

**Primary Technical Constraints:**
- **Limited GPU Memory**: Constrained batch sizes (1 per device) requiring gradient accumulation strategies
- **Sequence Length Limitations**: Maximum 256 tokens per essay, necessitating truncation for longer texts
- **Dataset Size**: 13,845 training samples requiring careful regularization to prevent overfitting
- **Evaluation Metric Alignment**: Quadratic Weighted Kappa (QWK) requires ordinal-aware loss functions rather than standard regression
- **Computational Budget**: Training transformer models with limited resources necessitated efficient mixed-precision training and early stopping

---

## How to Run the Project

This section provides step-by-step instructions for setting up and running the automated essay scoring project.

### Prerequisites

- **Python**: 3.11 to 3.14 (required by `pyproject.toml`)
- **Poetry**: Dependency management tool ([Installation Guide](https://python-poetry.org/docs/#installation))
- **GPU** (Recommended): CUDA-compatible GPU for transformer training (8GB+ VRAM recommended)
- **Kaggle Account**: For downloading the competition dataset

### Step 1: Clone the Repository

```bash
git clone https://github.com/suyog-pipliwaal/Automate-Essay-Scoring.git
cd Automate-Essay-Scoring
```

### Step 2: Install Dependencies

The project uses Poetry for dependency management. Install dependencies as follows:

```bash
# Install Poetry if not already installed
curl -sSL https://install.python-poetry.org | python3 -

# Install project dependencies
poetry install

# Activate the Poetry virtual environment
poetry shell
# Or alternatively: poetry env activate
```

**Note:** After activating the Poetry environment, you can run scripts directly. Alternatively, you can use `poetry run` prefix without activating the shell.

### Step 3: Download Dataset

The dataset should be downloaded from the [Kaggle Competition](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/) and placed in the `data/raw/` directory:

```bash
# Create data directory if it doesn't exist
mkdir -p data/raw

# Download train.csv and test.csv from Kaggle
# Place them in data/raw/
# Expected structure:
# data/raw/
#   ├── train.csv
#   ├── test.csv
#   └── sample_submission.csv
```

**Dataset Requirements:**
- `train.csv`: Must contain columns `essay_id`, `full_text`, and `score`
- `test.csv`: Must contain columns `essay_id` and `full_text`
- Files should be placed in `data/raw/` directory (as configured in `src/config/config.py`)

### Step 4: Run Baseline Models (Classical NLP)

Train and evaluate classical machine learning models using TF-IDF features:

```bash
# From project root directory (with Poetry environment activated)
poetry run python -m src.training.baseline
```

**Alternative (if Poetry shell is activated):**
```bash
python -m src.training.baseline
```

**What this does:**
- Preprocesses essays (contraction expansion, stopword removal, lemmatization)
- Converts text to TF-IDF vectors
- Trains multiple regression models (Linear, Ridge, Lasso, ElasticNet, SVR, Gradient Boosting)
- Selects best model based on QWK score
- Saves model and vectorizer to `src/models/exp_1_baseline/`
- Generates predictions and saves to `src/submission/exp_1_basline/submission.csv`

**Expected Output:**
- Model performance metrics for each algorithm
- Best model saved (typically Ridge Regression)
- Submission file generated

### Step 5: Run Transformer Models (BERT/DeBERTa)

Train transformer-based models with custom architecture:

```bash
# From project root directory (with Poetry environment activated)
poetry run python -m src.training.bert
```

**Alternative (if Poetry shell is activated):**
```bash
python -m src.training.bert
```

**Configuration:**
The training configuration is defined in `src/config/config.py`. Key parameters can be modified:

```python
@dataclass
class TrainingConfig:
  exp_name:str = "exp_deberta_prompt_head"  # Experiment name
  model_name:str = "microsoft/deberta-v3-base"  # or "bert-base-uncased"
  max_length:int = 256
  pooling:str = "attention"  # Options: "cls", "mean", "max", "attention"
  num_epoch:int = 5
  learning_rate:float = 2e-5
  # ... other hyperparameters
```

**What this does:**
- Loads and tokenizes training data
- Creates train/validation split (80/20, stratified by score)
- Initializes `BertForEssayScoring` model with specified configuration
- Trains using HuggingFace Trainer with:
  - Ordinal regression loss
  - Early stopping (patience=2)
  - Mixed precision training (FP16)
  - Gradient accumulation (effective batch size = 8)
- Saves best model to `src/models/{exp_name}/`
- Generates training curves and saves to `src/figures/{exp_name}/`

**Expected Output:**
- Rich console output with training progress
- Model checkpoints saved
- Training visualizations (loss curves, QWK over epochs)
- Final evaluation metrics

### Step 6: Customize Training Configuration

To run experiments with different configurations, modify `src/config/config.py`:

```python
@dataclass
class TrainingConfig:
  exp_name:str = "my_experiment"  # Change experiment name
  model_name:str = "bert-base-uncased"  # Switch model
  pooling:str = "mean"  # Try different pooling strategies
  num_epoch:int = 3  # Adjust epochs
  # ... modify other parameters
```

Or create a new config file and import it in `bert.py`.

### Step 7: Generate Predictions (Inference)

For baseline models, predictions are automatically generated during training. For transformer models, you can load a saved model and run inference:

```python
# Example inference script (create inference.py)
from transformers import AutoTokenizer, AutoModel
from src.training.bert import BertForEssayScoring
from src.data.load_data import load_test
from src.data.preprocess import tokenize_dataset
from datasets import Dataset

# Load model and tokenizer
model_path = "src/models/exp_deberta_prompt_head"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = BertForEssayScoring.from_pretrained(model_path)

# Load and tokenize test data
test_df = load_test()
test_ds = Dataset.from_pandas(test_df)
test_ds = tokenize_dataset(test_ds, tokenizer, max_length=256)
test_ds.set_format("torch")

# Generate predictions
# ... (implement prediction logic)
```

### Step 8: View Results

**Training Visualizations:**
- Loss curves: `src/figures/{exp_name}/train_vs_eval_loss.png`
- QWK progression: `src/figures/{exp_name}/eval_qwk.png`

**Model Checkpoints:**
- Baseline: `src/models/exp_1_baseline/`
- Transformer: `src/models/{exp_name}/`

**Submissions:**
- Baseline predictions: `src/submission/exp_1_basline/submission.csv`

### Troubleshooting

**Common Issues:**

1. **CUDA Out of Memory:**
   - Reduce `train_batch_size` to 1 (already set)
   - Increase `gradient_accumalation_steps` to maintain effective batch size
   - Enable gradient checkpointing (uncomment in `bert.py` line 203-204)
   - Reduce `max_length` (e.g., 128 instead of 256)

2. **NLTK Data Missing:**
   ```bash
   python -c "import nltk; nltk.download('wordnet'); nltk.download('stopwords')"
   ```

3. **Dataset Not Found:**
   - Ensure `train.csv` and `test.csv` are in `data/raw/`
   - Check file permissions and paths in `src/config/config.py`

4. **Poetry Installation Issues:**
   - Use `pip install poetry` as alternative
   - Or install dependencies manually from `pyproject.toml`

5. **Model Download Fails:**
   - Check internet connection
   - HuggingFace models are downloaded automatically on first use
   - For offline use, download models manually and specify local path

### Running on Different Hardware

**CPU-Only Training:**
- Set `fp16=False` in `TrainingConfig`
- Reduce `max_length` to 128
- Use smaller models (e.g., `distilbert-base-uncased`)

**Multi-GPU Training:**
- Modify `bert.py` to use `torch.nn.DataParallel` or `DistributedDataParallel`
- Adjust batch sizes accordingly

**Google Colab:**
- Upload project to Google Drive
- Install dependencies: `!pip install transformers torch datasets rich`
- Enable GPU runtime
- Adjust paths for Colab environment

### Project Structure Reference

```
Automate-Essay-Scoring/
├── data/raw/              # Dataset files (train.csv, test.csv)
├── src/
│   ├── config/           # Configuration files
│   ├── data/             # Data loading and preprocessing
│   ├── training/         # Training scripts
│   │   ├── baseline.py   # Classical ML models
│   │   ├── bert.py       # Transformer models
│   │   └── pooling.py    # Pooling strategies
│   ├── models/           # Saved model checkpoints
│   ├── figures/          # Training visualizations
│   ├── submission/       # Generated predictions
│   └── utils/            # Utility functions
├── pyproject.toml        # Dependencies and project config
└── REPORT.md             # This report
```

---

## Section 2: Technical Implementation (Detailed)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Input: Essay Text (full_text)                │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Tokenizer     │
                    │  (max_len=256) │
                    └────────┬───────┘
                             │
                             ▼
                    ┌────────────────┐
                    │  Transformer  │
                    │  Encoder      │
                    │  (BERT/DeBERTa)│
                    └────────┬───────┘
                             │
                    [Token Embeddings: seq_len × hidden_size]
                             │
                             ▼
                    ┌────────────────┐
                    │  Pooling Layer │
                    │  (CLS/Mean/    │
                    │   Max/Attention)│
                    └────────┬───────┘
                             │
                    [Document Embedding: hidden_size]
                             │
                             ▼
                    ┌────────────────┐
                    │  Shared Layer  │
                    │  Linear → GLU  │
                    │  → Dropout     │
                    └────────┬───────┘
                             │
                    [Shared Rep: hidden_size]
                             │
                             ▼
                    ┌────────────────┐
                    │ Prompt-Specific│
                    │  Heads (1-8)   │
                    └────────┬───────┘
                             │
                    [Logits: 5 ordinal thresholds]
                             │
                             ▼
                    ┌────────────────┐
                    │ Ordinal Loss   │
                    │ (BCE + Penalty)│
                    └────────────────┘
```

**Architecture Explanation:** The model processes essays through a pre-trained transformer encoder (BERT-base [Devlin et al., 2018] or DeBERTa-v3 [He et al., 2021]) to generate contextualized token embeddings, which are then aggregated into a single document representation using configurable pooling strategies (attention pooling [Yang et al., 2016] performed best). A shared representation layer with Gated Linear Unit (GLU) activation [Dauphin et al., 2017] processes the pooled embedding, which is then passed through prompt-specific linear heads that learn distinct scoring patterns for each of the 8 essay prompts, producing 5 ordinal logits that represent score thresholds for the 6-point scale.

### Code Walk-Through: Critical Function

The `ordinal_loss` function is critical because it addresses the fundamental challenge that essay scoring is an ordinal problem, not a standard regression task:

```python
def ordinal_loss(self, logits, labels):
    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels.float())
    penalty = torch.relu(logits[:, :-1] - logits[:, 1:]).mean()
    return bce + 0.1 * penalty
```

**Function Breakdown:**

1. **Input Processing**: 
   - `logits`: Shape `[batch_size, 5]` representing 5 binary thresholds (score > 1, > 2, > 3, > 4, > 5)
   - `labels`: Shape `[batch_size, 5]` binary labels converted from ordinal scores (e.g., score 4 → `[1, 1, 1, 1, 0]`)

2. **Binary Cross-Entropy Component**:
   - Treats each threshold as an independent binary classification problem
   - Uses sigmoid activation (via `binary_cross_entropy_with_logits`) to predict probability of score exceeding each threshold
   - This allows the model to learn ordinal relationships implicitly

3. **Monotonicity Penalty**:
   - `logits[:, :-1] - logits[:, 1:]` computes differences between consecutive thresholds
   - `torch.relu()` ensures only positive differences (violations) are penalized
   - Enforces that higher thresholds should have lower logits (since P(score > 5) < P(score > 4) < ... < P(score > 1))
   - Weight of 0.1 balances the BCE loss with the ordinal constraint

4. **Why This Matters**: Standard MSE loss treats score differences uniformly (predicting 1 vs 2 is same error as 5 vs 6), but QWK penalizes larger errors quadratically. This ordinal loss better aligns with the evaluation metric and respects the natural ordering of essay scores, following the principles of ordinal regression [Frank & Hall, 2001].

### Data Flow: Forward Pass Operation

**Step-by-Step Data Flow for a Single Essay:**

1. **Input Preparation**:
   ```
   Essay Text: "The importance of education cannot be overstated..."
   ↓
   Tokenizer: [101, 1996, 2651, 1997, 3698, ...]  # BERT token IDs
   Attention Mask: [1, 1, 1, 1, 1, ..., 0, 0, 0]  # 1 for real tokens, 0 for padding
   Prompt ID: 1  # Which essay prompt this belongs to
   ```

2. **Encoder Forward Pass**:
   ```
   input_ids [256] → Transformer Encoder → hidden_states [256, 768]
   # 256 tokens × 768-dimensional embeddings (BERT-base hidden size)
   ```

3. **Pooling Operation** (Attention Mode):
   ```python
   # From pooling.py, lines 24-32
   # Implements attention-based pooling (Yang et al., 2016)
   scores = self.attn(hidden_state).squeeze(-1)  # [256, 1] → [256]
   scores = scores.masked_fill(attention_mask == 0, min_val)  # Mask padding
   weights = torch.softmax(scores, dim=1).unsqueeze(-1)  # [256, 1]
   pooled = (hidden_state * weights).sum(dim=1)  # [256, 768] → [768]
   ```
   Result: `[768]` - Single document embedding weighted by learned attention

4. **Shared Representation**:
   ```
   pooled [768] → Linear(768 → 1536) → GLU → Dropout(0.2) → shared [768]
   # GLU: Gated Linear Unit (Dauphin et al., 2017) for selective information flow
   ```

5. **Prompt-Specific Head**:
   ```python
   # From bert.py, lines 152-154
   for pid in prompt_id.unique():
       mask = prompt_id == pid
       logits[mask] = self.prompt_heads[str(pid.item())](shared[mask])
   ```
   Result: `[5]` - Logits for 5 ordinal thresholds

6. **Loss Computation**:
   ```
   Labels (score=4): [1, 1, 1, 1, 0]  # Binary thresholds
   Logits: [-0.5, 0.3, 1.2, 2.1, -1.0]
   ↓
   BCE Loss: Binary cross-entropy between sigmoid(logits) and labels
   Penalty: ReLU([-0.8, -0.9, -0.9, 3.1]) = [0, 0, 0, 3.1] → mean = 0.775
   Final Loss: BCE + 0.1 × 0.775
   ```

7. **Prediction (Inference)**:
   ```python
   # From compute_metrics, lines 116-117
   probs = 1 / (1 + np.exp(-logits))  # Sigmoid: [0.38, 0.57, 0.77, 0.89, 0.27]
   preds = (probs > 0.5).sum(axis=1) + 1  # Count thresholds passed: 4 → Score = 4
   ```

### Classical NLP Pipeline: TF-IDF Vectorization (`to_vectors`)

The baseline approach uses a traditional NLP pipeline that converts preprocessed text into sparse numerical features suitable for classical machine learning models.

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│         Preprocessed Essays (clean_input column)            │
│  ["importance education cannot overstated", ...]            │
└────────────────────────┬────────────────────────────────────┘
                          │
                          ▼
              ┌───────────────────────┐
              │  Train/Test Split    │
              │  (80/20 stratified)  │
              └───────────┬───────────┘
                          │
        ┌─────────────────┴─────────────────┐
        │                                   │
        ▼                                   ▼
┌───────────────┐                  ┌───────────────┐
│ Training Set  │                  │  Test Set     │
│ (11,076)      │                  │  (2,769)      │
└───────┬───────┘                  └───────┬───────┘
        │                                   │
        │                                   │
        ▼                                   ▼
┌──────────────────────────────────────────────────┐
│      TF-IDF Vectorizer.fit_transform()          │
│  • Learn vocabulary from training set           │
│  • Compute IDF (Inverse Document Frequency)     │
│  • Transform training texts to TF-IDF vectors   │
│                                                  │
│  Parameters:                                    │
│  • sublinear_tf=True  (log scaling)             │
│  • max_df=0.5       (ignore >50% docs)         │
│  • min_df=5         (ignore <5 docs)            │
│  • stop_words="english"                         │
└───────────────────┬─────────────────────────────┘
                    │
        ┌───────────┴───────────┐
        │                       │
        ▼                       ▼
┌──────────────┐      ┌──────────────┐
│ X_train      │      │ X_test       │
│ (11,076, N)  │      │ (2,769, N)   │
│ Sparse Matrix│      │ Sparse Matrix│
│              │      │              │
│ N ≈ 5,000-   │      │ (vocab size) │
│    10,000    │      │              │
│ features     │      │              │
└──────┬───────┘      └──────┬───────┘
       │                     │
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────┐
│ y_train      │      │ y_test       │
│ [4, 5, 3, ...]│      │ [4, 6, 2, ...]│
│ Scores       │      │ Scores       │
└──────────────┘      └──────────────┘
```

#### Code Walk-Through: `to_vectors` Function

```python
@timeit
def to_vectors(dataset:pd.DataFrame):
  # Step 1: Split into train/test sets
  train, test = train_test_split(dataset, test_size=0.2)
  
  # Step 2: Initialize TF-IDF Vectorizer
  vectorizer = text.TfidfVectorizer(
      sublinear_tf=True,      # Apply 1 + log(tf) instead of raw tf
      max_df=0.5,             # Ignore terms in >50% of documents
      min_df=5,               # Ignore terms in <5 documents
      stop_words="english"    # Remove English stopwords
  )
  
  # Step 3: Fit on training data and transform
  X_train = vectorizer.fit_transform(train.clean_input.tolist())
  y_train = np.array(train.score.tolist())
  
  # Step 4: Transform test data (using learned vocabulary)
  X_test = vectorizer.transform(test.clean_input.tolist())
  y_test = np.array(test.score.tolist())
  
  # Step 5: Extract feature names and save vectorizer
  feature_name = vectorizer.get_feature_names_out()
  save_model(vectorizer, model_name="vectorize")
  
  return X_train, y_train, X_test, y_test, feature_name, vectorizer
```

**Function Breakdown:**

1. **Train/Test Split** (line 52):
   - 80/20 split ensures sufficient training data while maintaining a representative test set
   - Stratified splitting (by score) would be ideal but not implemented in baseline

2. **TF-IDF Vectorizer Initialization** (line 53):
   - **`sublinear_tf=True`**: Applies logarithmic scaling `1 + log(tf)` to term frequency, dampening the effect of very frequent words
   - **`max_df=0.5`**: Filters out terms appearing in more than 50% of documents (too common, low discriminative power)
   - **`min_df=5`**: Filters out terms appearing in fewer than 5 documents (too rare, likely noise or typos)
   - **`stop_words="english"`**: Removes common English stopwords (already done in preprocessing, but provides redundancy)

3. **Fit and Transform Training Data** (line 55):
   - **`fit_transform()`**: Two-step process:
     - **Fit**: Learns vocabulary from training corpus, computes IDF (Inverse Document Frequency) weights
     - **Transform**: Converts each essay to a sparse TF-IDF vector
   - **Output**: Sparse matrix of shape `[n_train_samples, vocabulary_size]`
   - Each row represents an essay as a high-dimensional sparse vector

4. **Transform Test Data** (line 59):
   - **`transform()`**: Uses the vocabulary learned from training (no fitting on test data)
   - Ensures test set uses same feature space as training
   - Terms not seen in training are ignored (out-of-vocabulary handling)

5. **TF-IDF Formula**:
   ```
   TF-IDF(t, d) = TF(t, d) × IDF(t)
   
   Where:
   - TF(t, d) = (1 + log(count of t in d))  [sublinear scaling]
   - IDF(t) = log(N / (1 + df(t)))
   - N = total number of documents
   - df(t) = number of documents containing term t
   ```

**Example Transformation:**

```
Input Essay: "The importance of education cannot be overstated. 
              Education provides knowledge and skills."

After Preprocessing: "importance education cannot overstated 
                      education provides knowledge skills"

TF-IDF Vector (sparse, showing only non-zero features):
  'education': 0.45    (appears twice, high TF)
  'importance': 0.32
  'knowledge': 0.28
  'overstated': 0.25
  'provides': 0.22
  'skills': 0.20
  ... (5000+ other features = 0)
```

**Key Characteristics:**
- **Sparse Representation**: Most features are zero (typical sparsity >99%)
- **High Dimensionality**: Vocabulary size typically 5,000-10,000 features
- **Memory Efficient**: Uses scipy sparse matrices (CSR format)
- **No Context**: Loses word order and semantic relationships

### BERT Tokenization Pipeline

The transformer approach uses subword tokenization (WordPiece [Schuster & Nakajima, 2012]) to handle vocabulary limitations and out-of-vocabulary words effectively, as implemented in BERT [Devlin et al., 2018].

#### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              Raw Essay Text (full_text)                      │
│  "The importance of education cannot be overstated..."     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │  AutoTokenizer       │
              │  (BERT/DeBERTa)      │
              │                      │
              │  Steps:              │
              │  1. Basic Tokenization│
              │  2. WordPiece Split  │
              │  3. Add Special Tokens│
              │  4. Truncate/Pad     │
              └──────────┬───────────┘
                         │
        ┌────────────────┴────────────────┐
        │                                   │
        ▼                                   ▼
┌──────────────────┐            ┌──────────────────┐
│ input_ids        │            │ attention_mask   │
│ [101, 1996, ...] │            │ [1, 1, 1, ...]  │
│                  │            │                  │
│ Shape: [256]     │            │ Shape: [256]     │
│                  │            │                  │
│ Special Tokens:  │            │ 1 = real token  │
│ • [CLS] = 101    │            │ 0 = padding      │
│ • [SEP] = 102    │            │                  │
│ • [PAD] = 0      │            │                  │
└──────────────────┘            └──────────────────┘
```

#### Code Walk-Through: `tokenize_dataset` Function

```python
from transformers import AutoTokenizer

def tokenize_dataset(dataset, tokenizer, max_length):
  def tokenize(batch):
    return tokenizer(
        batch['full_text'],           # Input text column
        truncation=True,              # Truncate if > max_length
        max_length=max_length,        # 256 tokens
        return_token_type_ids=False   # Not needed for single sequence
    )
  return dataset.map(tokenize, batched=True)
```

**Tokenization Process (Step-by-Step):**

1. **Basic Tokenization**:
   ```
   Input: "The importance of education cannot be overstated."
   
   Step 1 - Whitespace/Word Splitting:
   ["The", "importance", "of", "education", "cannot", "be", "overstated", "."]
   ```

2. **WordPiece Tokenization**:
   ```
   Step 2 - Subword Splitting (WordPiece algorithm):
   "importance" → ["import", "##ance"]      # Split if not in vocab
   "education" → ["education"]              # Keep whole if in vocab
   "overstated" → ["over", "##stated"]     # Split compound word
   "cannot" → ["can", "##not"]              # Split contraction
   
   Result: ["The", "import", "##ance", "of", "education", 
            "can", "##not", "be", "over", "##stated", "."]
   ```

3. **Add Special Tokens**:
   ```
   Step 3 - BERT Special Tokens:
   [CLS] + tokens + [SEP]
   
   ["[CLS]", "The", "import", "##ance", "of", "education", 
    "can", "##not", "be", "over", "##stated", ".", "[SEP]"]
   ```

4. **Convert to Token IDs**:
   ```
   Step 4 - Vocabulary Mapping:
   [CLS] → 101
   "The" → 1996
   "import" → 4567
   "##ance" → 8901
   ...
   [SEP] → 102
   
   Result: [101, 1996, 4567, 8901, 1997, 3698, 2064, 2365, 2022, 2088, 10172, 1012, 102]
   ```

5. **Truncation/Padding**:
   ```python
   # If sequence length < max_length (256):
   input_ids = [101, 1996, ..., 102, 0, 0, 0, ..., 0]  # Pad with 0
   attention_mask = [1, 1, ..., 1, 0, 0, 0, ..., 0]   # 0 for padding
   
   # If sequence length > max_length (256):
   input_ids = [101, 1996, ..., token_255, 102]  # Truncate, keep [CLS] and [SEP]
   attention_mask = [1, 1, ..., 1]               # All 1s (no padding)
   ```

**Tokenization Example:**

```python
# Example from bert.py, lines 183-186
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
train_ds = tokenize_dataset(train_ds, tokenizer, max_length=256)

# Input Essay:
essay = "The importance of education cannot be overstated. Education provides knowledge."

# Tokenization Output:
{
    'input_ids': [101, 1996, 2651, 1997, 3698, 2064, 2365, 2022, 2088, 10172, 1012, 
                  3698, 2975, 3137, 1012, 102, 0, 0, ..., 0],  # 256 total
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
                       0, 0, ..., 0],  # 16 real tokens + 240 padding
    'full_text': "The importance of education cannot be overstated. Education provides knowledge."
}
```

**Key Advantages of WordPiece Tokenization** [Schuster & Nakajima, 2012]:

1. **Handles OOV Words**: Unknown words are split into subwords that are likely in vocabulary
   - "overstated" → ["over", "##stated"] (both subwords likely in vocab)

2. **Compact Vocabulary**: ~30,000 tokens vs. millions for full word vocabularies

3. **Morphological Awareness**: Handles word variations naturally
   - "education", "educational", "educate" share common subwords

4. **Language Agnostic**: Works across languages with appropriate tokenizers

**Comparison: Classical vs. Transformer Tokenization**

| Aspect | TF-IDF (Classical) | WordPiece (BERT) |
|--------|-------------------|------------------|
| **Vocabulary** | Fixed from training corpus | Pre-trained (30K tokens) |
| **OOV Handling** | Ignored (lost) | Split into subwords |
| **Context** | None (bag-of-words) | Preserved (sequence) |
| **Dimensionality** | 5K-10K features | 256 tokens (fixed) |
| **Sparsity** | Very sparse (>99%) | Dense (all tokens used) |
| **Word Order** | Lost | Preserved |
| **Semantic Info** | None | Rich (contextual embeddings) |

---

## Section 3: Technical Decisions (Core)

### Decision 1: Attention-Based Pooling vs. CLS/Mean Pooling

**Choice**: Implemented and selected attention-based pooling over simpler alternatives (CLS token or mean pooling).

**Trade-offs:**

**Advantages:**
- **Dynamic Focus**: Learned attention weights allow the model to emphasize quality-indicating phrases (e.g., "well-structured argument", "clear thesis") while downweighting filler words, following the attentive pooling approach [Yang et al., 2016]
- **Better for Long Essays**: Unlike CLS pooling which assumes a single token captures all information, attention can aggregate information across the entire sequence
- **Interpretability**: Attention weights can be visualized to understand which parts of essays drive scoring decisions
- **Performance**: Achieved ~2-3% QWK improvement over mean pooling (0.78 vs 0.76), which is a standard baseline for sentence embeddings [Chen et al., 2018]

**Disadvantages:**
- **Additional Parameters**: Introduces a learnable linear layer (768 → 1) per model, increasing parameter count
- **Computational Overhead**: Requires computing attention scores and softmax for each token, adding O(seq_len) operations
- **Training Complexity**: More hyperparameters to tune (attention initialization, masking strategy)

**Rationale**: The performance gain (2-3% QWK) justified the minimal computational overhead, especially given that pooling occurs once per forward pass. The interpretability benefit was valuable for understanding model behavior.

### Decision 2: Ordinal Regression Loss vs. Mean Squared Error

**Choice**: Custom ordinal regression loss (BCE + monotonicity penalty) instead of standard MSE regression.

**Trade-offs:**

**Advantages:**
- **Metric Alignment**: QWK penalizes larger errors quadratically; ordinal loss naturally aligns with this by treating thresholds as ordered, following ordinal regression principles [Frank & Hall, 2001]
- **Better Generalization**: Respects the ordinal nature of scores (predicting 2 when true is 1 is better than predicting 6)
- **Interpretability**: Threshold-based predictions are more interpretable than continuous regression values
- **Performance**: Achieved ~1-2% QWK improvement over MSE (0.79 vs 0.77)

**Disadvantages:**
- **Implementation Complexity**: Requires custom loss function and label transformation (scores → binary thresholds)
- **Hyperparameter Tuning**: Penalty weight (0.1) requires tuning to balance BCE and ordinal constraint
- **Prediction Post-Processing**: Requires converting logits back to scores via threshold counting

**Rationale**: The ordinal nature of essay scores is fundamental to the problem—scores represent ordered quality levels, not arbitrary numeric values. The performance improvement and better alignment with QWK made this the clear choice despite added complexity.

### Scaling Bottleneck: GPU Memory Constraints

**Bottleneck**: Limited GPU memory (typically 8-16GB) prevented training with reasonable batch sizes. Standard transformer training requires batch_size ≥ 8-16 for stable gradients, but loading even batch_size=2 with sequence_length=256 and hidden_size=768 exceeded available memory.

**Mitigation Strategy:**

1. **Gradient Accumulation**:
   ```python
   train_batch_size = 1  # Process 1 sample at a time
   gradient_accumulation_steps = 8  # Accumulate over 8 steps
   # Effective batch size = 1 × 8 = 8
   ```
   - Processes samples sequentially but accumulates gradients before updating weights
   - Achieves same effective batch size with 8× less memory

2. **Mixed Precision Training (FP16)**:
   ```python
   fp16 = True  # Use 16-bit floats instead of 32-bit
   ```
   - Reduces memory usage by ~50% with minimal accuracy loss
   - Automatic loss scaling prevents gradient underflow

3. **Dynamic Padding**:
   ```python
   data_collator = DataCollatorWithPadding(tokenizer)
   # Pads each batch to its longest sequence, not global max_length
   ```
   - Reduces average sequence length per batch
   - Saves memory especially when many essays are shorter than 256 tokens

4. **Model Checkpointing** (commented but available):
   ```python
   # model.encoder.gradient_checkpointing_enable()
   # model.encoder.config.use_cache = False
   ```
   - Trades computation for memory by recomputing activations during backward pass
   - Can reduce memory by ~40% at cost of ~20% slower training

**Result**: Successfully trained DeBERTa-v3 (184M parameters) on consumer GPUs with effective batch size of 8, enabling experimentation without cloud compute resources.

---

## Section 4: Learning & Iteration (Concise)

### Technical Mistake: Initial Over-Reliance on CLS Token Pooling

**Mistake**: Initially used CLS token pooling assuming the pre-trained BERT model would encode all necessary essay information in the `[CLS]` token, following common BERT fine-tuning practices.

**What Happened**: 
- Achieved QWK of ~0.76, which was competitive but plateaued quickly
- Model struggled with longer essays (>200 tokens) where important information appeared mid-sequence
- Attention visualization showed the model was not effectively utilizing information beyond the first 50-100 tokens

**What I Learned**:
- **Pre-trained models don't guarantee optimal representations**: While CLS pooling works well for short classification tasks, long-form text (essays) requires explicit aggregation strategies
- **Ablation studies are essential**: Systematically testing pooling strategies (CLS, mean, max, attention) revealed that attention pooling provided 2-3% improvement, which is significant in competitive settings
- **Task-specific architecture matters**: What works for general NLP tasks may not be optimal for domain-specific problems like essay scoring

**Impact**: This mistake led to the implementation of a flexible pooling module (`pooling.py`) that supports multiple strategies, enabling systematic experimentation and ultimately improving final QWK from 0.76 to 0.80+.

### What I Would Do Differently Today

**Hierarchical Processing for Long Essays**: Instead of truncating essays to 256 tokens, I would implement a hierarchical approach that processes essays in overlapping chunks and aggregates representations.

**Rationale**: 
- Current approach loses information from essays longer than 256 tokens (approximately 200 words)
- Many high-quality essays exceed this length, and important arguments often appear in later paragraphs
- Hierarchical processing (e.g., sentence-level → paragraph-level → document-level) would preserve full essay content

**Implementation Approach**:
1. Split essays into sentence or paragraph chunks (each ≤ 256 tokens)
2. Process each chunk through the encoder independently
3. Aggregate chunk representations using attention or LSTM
4. Pass aggregated representation through the same scoring head

**Expected Benefits**:
- Handle essays of any length without information loss
- Better capture discourse structure and argument flow
- Potential 1-2% additional QWK improvement for longer essays

**Trade-off**: Increased computational cost (processing multiple chunks per essay) and added architectural complexity, but the performance gain would likely justify it for production systems.

---

## Additional Context: Dataset and Baseline Results

### Dataset Overview

- **Total Samples:** 17,307 essays
- **Training Samples:** 13,845 essays
- **Test Samples:** 3,462 essays
- **Evaluation Metric:** Quadratic Weighted Kappa (QWK)
- **Score Range:** 1-6 (ordinal scale)

### Baseline Performance

| Approach                    | Model                    | Pooling    | QWK Score | Notes                          |
| --------------------------- | ------------------------ | ---------- | --------- | ------------------------------ |
| Baseline                    | Ridge Regression         | N/A        | 0.7044    | TF-IDF features                |
| Transformer Baseline        | BERT-base                | Mean       | ~0.76     | Initial transformer approach   |
| Improved Pooling            | BERT-base                | Attention  | ~0.78     | Dynamic attention weighting    |
| Ordinal Regression          | BERT-base                | Attention  | ~0.79     | Ordinal loss alignment        |
| Advanced Architecture       | DeBERTa-v3 + Prompt Heads | Attention | **~0.80+** | Best performing configuration |

---

## Methodology Details

### Baseline Approach: Classical NLP

The project began with traditional NLP pipelines using TF-IDF vectorization and multiple regression models:

**Preprocessing Pipeline:**
- Contraction expansion (e.g., "don't" → "do not")
- Stopword removal
- Lemmatization using NLTK WordNet
- Text cleaning and normalization

**TF-IDF Configuration:**
- Sublinear TF scaling
- Max document frequency: 0.5
- Min document frequency: 5

**Model Comparison:**
- **Ridge Regression** (α=1.0): Best baseline with QWK 0.7044
- Linear Regression: Severe overfitting (train: 0.9750, test: 0.4041)
- SVR: Competitive (0.6994) but computationally expensive
- Gradient Boosting: Moderate performance (0.6426)

### Transformer Architecture Components

**1. Encoder**: Pre-trained transformer (BERT-base or DeBERTa-v3)
- Generates contextualized token embeddings
- Handles variable-length sequences with attention masks

**2. Pooling Strategies** (implemented in `pooling.py`):
- **CLS**: Uses `[CLS]` token embedding (standard BERT approach [Devlin et al., 2018])
- **Mean**: Averages all token embeddings (common baseline for sentence embeddings [Chen et al., 2018])
- **Max**: Maximum value per dimension
- **Attention**: Learned attention-weighted aggregation [Yang et al., 2016] (best performer)

**3. Shared Representation Layer**:
- Linear transformation: `hidden_size → hidden_size × 2`
- Gated Linear Unit (GLU) activation [Dauphin et al., 2017]
- Dropout regularization (0.2)

**4. Prompt-Specific Heads**:
- Separate linear layers for each of 8 essay prompts
- Allows learning prompt-specific scoring patterns
- Shared encoder ensures transfer of general language understanding

**5. Ordinal Regression Output**:
- Produces 5 logits representing thresholds: `[score > 1, > 2, > 3, > 4, > 5]`
- Custom loss: `BCE + 0.1 × monotonicity_penalty` (based on ordinal regression framework [Frank & Hall, 2001])

### Training Configuration

**Hyperparameters:**
- Learning Rate: 2e-5
- Batch Size: 1 per device (effective: 8 with gradient accumulation)
- Epochs: 3-5 (early stopping patience: 2)
- Weight Decay: 0.01
- Max Gradient Norm: 1.0
- Mixed Precision: FP16 enabled
- Max Sequence Length: 256 tokens

**Training Strategies:**
- Stratified train/validation splits
- Early stopping on validation QWK
- Best model selection and persistence
- Automated training curve visualization

---

## Code Organization

```
src/
├── config/          # Configuration management (Pydantic/dataclasses)
├── data/            # Data loading and preprocessing
├── models/          # Saved model checkpoints
├── training/        # Training scripts (baseline, BERT, pooling)
├── utils/           # Utilities (kappa, plots, seed, contractions)
└── figures/         # Training visualizations
```

**Key Design Patterns:**
- Centralized configuration management
- Reproducibility through seed setting
- Rich console output for training progress
- Automated visualization of training metrics
- Systematic model and tokenizer persistence

---

## Future Directions

1. **Hierarchical Models**: Process long essays in chunks and aggregate representations
2. **Multi-Task Learning**: Jointly learn related tasks (coherence, grammar, argumentation)
3. **Ensemble Methods**: Combine predictions from multiple models
4. **Advanced Architectures**: Explore Longformer, BigBird for long sequences
5. **Data Augmentation**: Increase training data diversity
6. **Explainability**: Understand which essay features drive scoring decisions
7. **Cross-Prompt Transfer**: Better leverage patterns across different essay prompts

---

## References

- **Kaggle Competition**: [Learning Agency Lab – Automated Essay Scoring 2.0](https://www.kaggle.com/competitions/learning-agency-lab-automated-essay-scoring-2/)

- **BERT**: Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805). *arXiv preprint arXiv:1810.04805*.

- **DeBERTa**: He, P., Liu, X., Gao, J., & Chen, W. (2021). [DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654). *ICLR 2021*.

- **Attention Pooling**: Yang, Z., Yang, D., Dyer, C., He, X., Smola, A., & Hovy, E. (2016). [Hierarchical Attention Networks for Document Classification](https://aclanthology.org/N16-1174/). *NAACL 2016*. (Also: Santos, C. N. dos, Gatti, M. (2016). [Attentive Pooling Networks](https://arxiv.org/abs/1602.03609). *arXiv preprint arXiv:1602.03609*.)

- **Mean Pooling & Generalized Pooling**: Chen, Q., Ling, Z. H., & Zhu, X. (2018). [Enhancing Sentence Embedding with Generalized Pooling](https://aclanthology.org/C18-1154/). *COLING 2018*.

- **Ordinal Regression**: Frank, E., & Hall, M. (2001). [A Simple Approach to Ordinal Classification](https://link.springer.com/chapter/10.1007/3-540-44795-4_13). *ECML 2001*.

- **Gated Linear Unit (GLU)**: Dauphin, Y. N., Fan, A., Auli, M., & Grangier, D. (2017). [Language Modeling with Gated Convolutional Networks](https://arxiv.org/abs/1612.08083). *ICML 2017*.

- **WordPiece Tokenization**: Schuster, M., & Nakajima, K. (2012). [Japanese and Korean Voice Search](https://ieeexplore.ieee.org/document/6289079). *ICASSP 2012*.

- **HuggingFace Transformers**: [Documentation](https://huggingface.co/docs/transformers)

---

*Report generated from project codebase analysis and experimental results.*
