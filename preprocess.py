import pandas as pd
import os
import pickle
from sklearn.model_selection import train_test_split

# 1. Trova la cartella 'src' (dove si trova questo script .py)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Definisci la cartella 'data' tornando indietro di un livello (..)
DATA_DIR = os.path.join(SRC_DIR, '..', 'data')

# 3. Crea il percorso esatto per il dataset originale
input_file = os.path.join(DATA_DIR, 'dataset.csv')

print("Caricamento dataset...")
df = pd.read_csv(input_file, sep=';', decimal=',')

if 'Customer_ID' in df.columns:
    df = df.drop('Customer_ID', axis=1)

X = df.drop('churn', axis=1)
y = df['churn']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_train = X_train.copy()
X_test = X_test.copy()


#ora categorizzo i dati e divido numeri da categorie(lettere)
num_cols = X_train.select_dtypes(
    include=['int64', 'float64', 'int32', 'float32']
).columns

cat_cols = X_train.select_dtypes(include=['object']).columns


# Salvataggio dei set NON preprocessati.
# Imputazione, encoding e scaling verranno fatti dentro le pipeline di training,
# così la GridSearchCV non avrà leakage sulle validation fold.

# Uso DATA_DIR per salvare tutto ordinatamente nella cartella dei dati
X_train.to_csv(os.path.join(DATA_DIR, 'X_train.csv'), index=False)
X_test.to_csv(os.path.join(DATA_DIR, 'X_test.csv'), index=False)
y_train.to_csv(os.path.join(DATA_DIR, 'y_train.csv'), index=False)
y_test.to_csv(os.path.join(DATA_DIR, 'y_test.csv'), index=False)

preprocessing_info = {
    'num_cols': list(num_cols),
    'cat_cols': list(cat_cols)
}

with open(os.path.join(DATA_DIR, 'preprocessing_info.pkl'), 'wb') as f:
    pickle.dump(preprocessing_info, f)

print("Split completato.")
print("File train/test grezzi e preprocessing_info salvati nella cartella 'data'.")