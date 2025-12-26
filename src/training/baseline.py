import re
import nltk
import numpy as np
import pandas as pd
import os
import joblib
from sklearn.feature_extraction import  text
from src.data.load_data import load_train
from src.utils.timeit import timeit
from src.utils.contractions import contractions
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from src.utils.kappa import quadratic_weighted_kappa
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

nltk.download('wordnet')
np.random.seed(42)

def clean_text(text:str, remove_stopword=True) -> str:
  text = text.split()
  new_text = []
  for words in text:
    if words in contractions:
      new_text.append(contractions[words])
    else:
      new_text.append(words)
  text = " ".join(new_text)
  text = re.sub(r'\'', ' ', text)
  if remove_stopword:
    text = text.split()
    stops = set(nltk.corpus.stopwords.words("english"))
    text = [w for w in text if w not in stops]
    text = " ".join(text)
  # text = nltk.WordPunctTokenizer().tokenize(text)
  return text
  
    
  


@timeit
def preprocessing(dataset:pd.DataFrame) -> pd.DataFrame:
  dataset['clean_input'] = list(map(clean_text, dataset.full_text))
  lemm = nltk.stem.WordNetLemmatizer()
  dataset['lemm_text'] =  list(map(lambda word:list(map(lemm.lemmatize, word)),dataset.clean_input))
  print(dataset.head())
  return dataset

@timeit
def to_vectors(dataset:pd.DataFrame):
  train, test = train_test_split(dataset, test_size=0.2)
  vectorizer = text.TfidfVectorizer(sublinear_tf=True, max_df=0.5, min_df=5, stop_words="english")
  
  X_train = vectorizer.fit_transform(train.clean_input.tolist())
  y_train = np.array(train.score.tolist())
  

  X_test = vectorizer.transform(test.clean_input.tolist())
  y_test = np.array(test.score.tolist())
  

  feature_name = vectorizer.get_feature_names_out()
  
  print(f"{len(dataset.score)} documents ")
  print(f"n_samples in training: {X_train.shape[0]}, n_features: {X_train.shape[1]}")
  print(f"n_samples in testing: {X_test.shape[0]}, n_features: {X_test.shape[1]}")
  
  return X_train, y_train, X_test, y_test, feature_name, vectorizer
  

def regression(traning_model, X_train, y_train, X_test, y_test):
  model = traning_model.fit(X_train, y_train)
  train_pred = model.predict(X_train)
  test_pred = model.predict(X_test)
  
  train_qwk = quadratic_weighted_kappa(y_train, train_pred)
  test_qwk  = quadratic_weighted_kappa(y_test, test_pred)
  print("Train QWK:", train_qwk)
  print("Test QWK:", test_qwk)
  return test_qwk

def save_model(model, model_name, path="src/models/exp_1_baseline/"):
  os.makedirs(path, exist_ok=True)
  full_path = os.path.join(path, f"{model_name}.pkl")
  print(f"saving best model {model_name} at {full_path}")
  print(full_path)
  joblib.dump(model, full_path)
if __name__ ==  '__main__':
  train_dataset = load_train()
  dataset = preprocessing(train_dataset)
  X_train, y_train, X_test, y_test, feature_name, vectorizer= to_vectors(dataset)
  # regression(X_train, y_train, X_test, y_test)
  models = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "lasso": Lasso(alpha=0.001),
        "elasticnet": ElasticNet(alpha=0.001, l1_ratio=0.5),
        "svr": SVR(C=1.0, epsilon=0.1),
        "gbr": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42
        ),
  }
  best_qwk = -1
  best_model = None
  best_name = None
  for name, model in models.items():
    print(f"Training model {name}")
    test_qwk = regression(model, X_train, y_train, X_test, y_test)
    if test_qwk > best_qwk:
      best_qwk = test_qwk
      best_model = model
      best_name = name
  save_model(best_model,best_name)
    
    
    