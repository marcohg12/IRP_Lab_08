# --------------------------------------------------------------------------------------------
# Este archivo contiene funciones para la clasificación de datos, formateo de datos,
# preprocesamiento de nuevos datos, carga de objetos de entrenamiento y 
# comprobación de la precisión del modelo generado
# --------------------------------------------------------------------------------------------

import pandas as pd
import joblib
import os
from constants import TRAINING_OBJS_DIR
from sklearn.metrics import accuracy_score

label_encoder = None
scaler = None
pca = None
model = None
imputer = None

# Carga los objetos de entrenamiento
# También carga el modelo de reconocimiento
def load_training_objs():

    global ordinal_encoder, scaler, pca, model, imputer

    ordinal_encoder = joblib.load(os.path.join(TRAINING_OBJS_DIR, 'ordinal_encoder.pkl'))
    scaler = joblib.load(os.path.join(TRAINING_OBJS_DIR, 'scaler.pkl'))
    pca = joblib.load(os.path.join(TRAINING_OBJS_DIR, 'pca.pkl'))
    model = joblib.load(os.path.join(TRAINING_OBJS_DIR, 'logistic_regression_model.pkl'))
    imputer = joblib.load(os.path.join(TRAINING_OBJS_DIR, 'imputer.pkl'))


# Recibe los 13 datos de una persona
# Formatea los datos en un objeto DataFrame de Pandas
# Retorna este objeto DataFrame
def format_data(age, workclass, education, education_num, marital_status, occupation, 
            relationship, race, sex, capital_gain, capital_loss, hours_per_week, native_country):
    
    data = pd.DataFrame({
        'age': [age],
        'workclass': [workclass],
        'education': [education],
        'education-num': [education_num],
        'marital-status': [marital_status],
        'occupation': [occupation],
        'relationship': [relationship],
        'race': [race],
        'sex': [sex],
        'capital-gain': [capital_gain],
        'capital-loss': [capital_loss],
        'hours-per-week': [hours_per_week],
        'native-country': [native_country]
    })

    return data

# Aplica el preprocesamiento al dataframe
# Utiliza los mismos objetos que se usaron en el entrenamiento
def preprocess_data(dataframe):

    global ordinal_encoder, scaler, pca

    data_encoded = ordinal_encoder.transform(dataframe)
    data_imputed  = imputer.transform(data_encoded)
    data_scaled = scaler.transform(data_imputed)
    data_pca = pca.transform(data_scaled)

    return data_pca

# Recibe un dataframe reducido con PCA (se aplicó la función preprocess_data anteriormente)
def classify(data):

    global model

    data_pred = model.predict(data)
    return data_pred

# Recibe un conjunto de datos de prueba y un conjunto de clasificaciones esperadas
# Retorna la tasa de aciertos del modelo
def get_model_accuracy(X_test, y_test):

    global model

    X_reduced = preprocess_data(X_test)
    y_pred = model.predict(X_reduced)
    accuracy = accuracy_score(y_test, y_pred)

    return accuracy