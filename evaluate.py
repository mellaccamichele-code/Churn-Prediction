import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def run_evaluation():
    print("Caricamento modelli e dati di test...")

    # 1. Impostazione dei percorsi aziendali (MLOps)
    SRC_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SRC_DIR, '..', 'data')
    MODELS_DIR = os.path.join(SRC_DIR, '..', 'models')

    # Caricamento modelli dalla cartella 'models'
    with open(os.path.join(MODELS_DIR, 'model_xgb.pkl'), 'rb') as f:
        model_xgb = pickle.load(f)

    with open(os.path.join(MODELS_DIR, 'model_lr.pkl'), 'rb') as f:
        model_lr = pickle.load(f)

    # Caricamento dati di test grezzi dalla cartella 'data'
    X_test = pd.read_csv(os.path.join(DATA_DIR, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(DATA_DIR, 'y_test.csv')).values.ravel()

    print("\n=========================================")
    print("  PERFORMANCE REPORT: XGBOOST")
    print("=========================================")

    # model_xgb è una pipeline completa:
    # preprocessor + XGBoost
    y_pred_xgb = model_xgb.predict(X_test)
    print(classification_report(y_test, y_pred_xgb))

    print("\n=========================================")
    print("  PERFORMANCE REPORT: LOGISTIC REGRESSION")
    print("=========================================")

    # model_lr è una pipeline completa:
    # preprocessor + scaler + Logistic Regression
    y_pred_lr = model_lr.predict(X_test)
    print(classification_report(y_test, y_pred_lr))

    # =================================================================
    # GRAFICO FEATURE IMPORTANCE
    # =================================================================
    print("\n--- GRAFICO FEATURE IMPORTANCE ---\n")

    # Ora model_xgb è una Pipeline, quindi il modello XGBoost vero
    # si recupera con named_steps['model']
    xgb_model = model_xgb.named_steps['model']

    importances = xgb_model.feature_importances_

    # Le feature prodotte dal ColumnTransformer sono salvate nella cartella 'models'
    with open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)

    feature_imp_df = pd.DataFrame({
        'Variabile': feature_names,
        'Importanza': importances
    })

    feature_imp_df = feature_imp_df.sort_values(
        by='Importanza',
        ascending=False
    )

    top_10 = feature_imp_df.head(10).copy()

    mappa_legenda_business = {
        'eqpdays': 'Device Age (Days)',
        'change_mou': '% Change in Monthly Minutes',
        'mou_Mean': 'Mean Monthly Minutes of Use',
        'months': 'Customer Tenure (Months)',
        'totmrc_Mean': 'Mean Total Recurring Charge',
        'rev_Mean': 'Mean Monthly Revenue',
        'hnd_price': 'Current Handset Price',
        'direct_Mean': 'Mean Direct Directory Calls',
        'ovrrev_Mean': 'Mean Overage Revenue',
        'uniqsubs': 'Unique Subscribers in Household',
        'avg3rev': 'Average Monthly Revenue Over Previous Three Months',
        'avgqty': 'Average Monthly Number of Calls Over Previous Three Months',
        'vceovr_Mean': 'Mean Revenue of Voice Overage',
        'crclscod': 'Credit Class Code'
    }

    def pulisci_nome_feature_finale(nome_grezzo):
        nome = str(nome_grezzo).strip()
        nome_pulito_base = nome.replace('text_format', '').replace('check', '')
        nome_lower = nome_pulito_base.lower()

        if nome in mappa_legenda_business:
            return mappa_legenda_business[nome]

        if nome_pulito_base in mappa_legenda_business:
            return mappa_legenda_business[nome_pulito_base]

        if nome_lower in mappa_legenda_business:
            return mappa_legenda_business[nome_lower]

        if 'refurb' in nome_lower:
            if '_r' in nome_lower or nome_pulito_base.lower().endswith('r'):
                return "Handset: Refurbished"
            if '_n' in nome_lower or nome_pulito_base.lower().endswith('n'):
                return "Handset: New"
            return "Handset: Refurbished vs New"

        if 'asl' in nome_lower:
            if '_y' in nome_lower or nome_pulito_base.lower().endswith('y'):
                return "Account Spending Limit: Yes"
            if '_n' in nome_lower or nome_pulito_base.lower().endswith('n'):
                return "Account Spending Limit: No"
            return "Account Spending Limit Status"

        if 'area' in nome_lower:
            regione = (
                nome_pulito_base
                .replace('area', '')
                .replace('AREA', '')
                .replace('_', ' ')
                .strip()
            )

            if not regione:
                return "Geographic Region"

            return f"Region: {regione.title()}"

        if 'ethnic' in nome_lower:
            pulito = (
                nome_pulito_base
                .replace('ethnic', '')
                .replace('_', '')
                .strip()
                .upper()
            )

            if pulito and len(pulito) == 1:
                return f"Ethnicity Roll-up Group {pulito}"

            return "Customer Ethnicity Group"

        if 'marital' in nome_lower:
            pulito = (
                nome_pulito_base
                .replace('marital', '')
                .replace('_', '')
                .strip()
                .upper()
            )

            if 'S' in pulito:
                return "Marital Status: Single"

            if 'M' in pulito:
                return "Marital Status: Married"

            return "Marital Status"

        return nome_pulito_base.replace('_', ' ').strip().title()

    top_10['Business_Key'] = top_10['Variabile'].apply(
        pulisci_nome_feature_finale
    )

    print("Top 10 feature più importanti:\n")
    print(
        top_10[['Variabile', 'Business_Key', 'Importanza']]
        .to_string(index=False)
    )

    plt.figure(figsize=(12, 6))
    sns.set_theme(style="whitegrid")

    sns.barplot(
        x='Importanza',
        y='Business_Key',
        data=top_10,
        color='#11caa0'
    )

    plt.title(
        "Top 10 Churn Drivers - XGBoost Feature Importance",
        fontsize=16,
        fontweight='bold',
        pad=15
    )

    plt.xlabel("Relative Importance", fontsize=12)
    plt.ylabel("Predictive Feature", fontsize=12)

    plt.tight_layout()

    # Salvataggio del grafico nella cartella 'models'
    output_file = os.path.join(MODELS_DIR, 'feature_importance_business.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.show()

    print(f"\nGrafico salvato come: {output_file}")


run_evaluation()