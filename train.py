import pandas as pd
import pickle
import os
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline #mi aiuta a costruire pipeline per xgboost e regr
from sklearn.compose import ColumnTransformer
from category_encoders import TargetEncoder

def run_training():
    # 1. Impostazione dei percorsi aziendali (MLOps)
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
    MODELS_DIR = os.path.join(SRC_DIR, '..', 'models')
    
    # Assicurati che la cartella models esista, altrimenti la crea
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("Caricamento dataset grezzi...")

    # Legge dalla cartella DATA_DIR
    X_train = pd.read_csv(os.path.join(DATA_DIR, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(DATA_DIR, 'y_train.csv')).values.ravel()

    with open(os.path.join(DATA_DIR, 'preprocessing_info.pkl'), 'rb') as f:
        preprocessing_info = pickle.load(f)

    num_cols = preprocessing_info['num_cols'] #la lista di tutte le colonne con i numeri (es. ['Device_Age', 'Mean_Monthly_Minutes', ...])
    cat_cols = preprocessing_info['cat_cols']

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')) #mediana nelle colonne numeriche nei missing values
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')), #moda per le variabili categoriche
        ('target_encoder', TargetEncoder()) #non ho fatto dummy per la maledizione della dimensionalità, target encoder sostituisce 
        #la variabile categorica con la sua frequenza nel dataset
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, num_cols),#('Nome_che_vuoi_tu', Cosa_deve_fare, Su_quali_colonne)
            ('cat', categorical_transformer, cat_cols)
        ]
    )

    # --- XGBOOST ---
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) #stratified controlla frequenza dei churn uguali nei fold

    print("Avvio Grid Search XGBoost...")

    xgb_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBClassifier(
            eval_metric='logloss',
            random_state=42,
            n_jobs=1
        ))
    ])

    param_grid_xgb = {
        'model__max_depth': [5, 7],
        'model__learning_rate': [0.03, 0.05],
        'model__n_estimators': [100, 200],
        'model__subsample': [0.8, 1.0], #80% di righe, riduciamo la casualità del modello
        'model__colsample_bytree': [0.8, 1.0] #80% colonne, magari alcune feature oscurano tutte le altre
    }

    grid_search_xgb = GridSearchCV(
        estimator=xgb_pipeline,
        param_grid=param_grid_xgb,
        cv=cv,
        scoring='f1',
        n_jobs=1,
        verbose=1
    )

    grid_search_xgb.fit(X_train, y_train) #qui per ogni setting di iperpar fa il fold a giro usando tutti e 5 come validation fa la media di F1
    #e ogni iterazione fa il preprocess dei 4 fold separati da quello di validation

    model_xgb = grid_search_xgb.best_estimator_

    print(f"Migliori parametri XGBoost: {grid_search_xgb.best_params_}")
    print(f"Miglior F1 medio XGBoost in cross-validation: {grid_search_xgb.best_score_:.4f}")

    # --- LOGISTIC REGRESSION ---
    print("\nAvvio Grid Search Logistic Regression...")

    lr_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('scaler', StandardScaler()), #standard scaler dopo il preprocess per standardizzare magnitudine feature
        ('model', LogisticRegression(
            max_iter=5000,
            random_state=42,
            solver='liblinear'
        ))
    ])

    param_grid_lr = {
        'model__C': [0.01, 0.1, 1, 10], #controlla la forza della regolarizzazione c bassi = tieni i coefficienti bassi
        'model__penalty': ['l2'], #regoralizzazione ridge, rimpicciolisce i coefficienti
        'model__class_weight': [None, 'balanced'] #aiuta la recall, penalizza gli errori sui clienti che hanno fatto churn
    }

    grid_search_lr = GridSearchCV(
        estimator=lr_pipeline,
        param_grid=param_grid_lr,
        cv=cv,
        scoring='f1',
        n_jobs=1, #ottimizza calcolo con CPU
        verbose=1 #barra di caricamento
    )

    grid_search_lr.fit(X_train, y_train)

    model_lr = grid_search_lr.best_estimator_

    print(f"Migliori parametri Logistic Regression: {grid_search_lr.best_params_}")
    print(f"Miglior F1 medio Logistic Regression in cross-validation: {grid_search_lr.best_score_:.4f}")

    # Salvataggio feature names (nei MODELS, perché serviranno per valutare il modello)
    feature_names = list(num_cols) + list(cat_cols)

    with open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_names, f)

    # Salvataggio modelli nella cartella MODELS
    with open(os.path.join(MODELS_DIR, 'model_xgb.pkl'), 'wb') as f:
        pickle.dump(model_xgb, f)

    with open(os.path.join(MODELS_DIR, 'model_lr.pkl'), 'wb') as f:
        pickle.dump(model_lr, f)

    print("\n Modelli salvati correttamente nella cartella 'models'.")

if __name__ == "__main__":
    run_training()