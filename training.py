# --------------------------------------------------------------------------------------------
# Este archivo contiene una función para entrenar un modelo supervisado usando componentes
# PCA, la función guarda los objetos usados en el entrenamiento
# --------------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import numpy as np
import joblib
import os
import time
from constants import TRAINING_OBJS_DIR


# Recibe el dataset, el nombre de la variable que se desea predecir y la cantidad de componentes
# Entrena un modelo utilizando PCA
# Retorna el dataset reducido utilizado para el entrenamiento
def train_dataset(X_train, Y_train, n_components):

    start_time = time.time()
    
    # Codificamos las variables cualitativas
    ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = np.nan)
    X_encoded = ordinal_encoder.fit_transform(X_train)

    # Reemplazamos datos faltantes por la media
    imputer = SimpleImputer(strategy = 'mean')
    X_imputed = imputer.fit_transform(X_encoded)
    
    # Estandarizamos los datos
    scaler = StandardScaler(with_mean = False)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Ejecutamos un PCA para reducir la dimensionalidad
    pca = PCA(n_components = n_components)
    X_pca = pca.fit_transform(X_scaled)
    
    # Entrenamos un modelo reconocedor
    model = RandomForestClassifier()
    model.fit(X_pca, Y_train)

    # Guardamos los objetos de preprocesamiento y el modelo para su uso posterior
    joblib.dump(ordinal_encoder, os.path.join(TRAINING_OBJS_DIR, 'ordinal_encoder.pkl'))
    joblib.dump(scaler, os.path.join(TRAINING_OBJS_DIR, 'scaler.pkl'))
    joblib.dump(pca, os.path.join(TRAINING_OBJS_DIR, 'pca.pkl'))
    joblib.dump(model, os.path.join(TRAINING_OBJS_DIR, 'logistic_regression_model.pkl'))
    joblib.dump(imputer, os.path.join(TRAINING_OBJS_DIR, 'imputer.pkl'))

    print(f"Tiempo de ejecución con {n_components} componentes: ", time.time() - start_time)

    return X_pca

# Prueba el proceso de entrenamiento y clasificación sin PCA
def test_process_with_no_pca(X_train, Y_train, X_test, Y_test):

    start_time = time.time()

    # Codificamos las variables cualitativas
    ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = np.nan)
    X_encoded = ordinal_encoder.fit_transform(X_train)

    # Reemplazamos datos faltantes por la media
    imputer = SimpleImputer(strategy = 'mean')
    X_imputed = imputer.fit_transform(X_encoded)
    
    # Estandarizamos los datos
    scaler = StandardScaler(with_mean = False)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Entrenamos un modelo reconocedor
    model = RandomForestClassifier()
    model.fit(X_scaled, Y_train)

    # Preprocesamos los datos de entrenamiento
    data_encoded = ordinal_encoder.transform(X_test)
    data_imputed  = imputer.transform(data_encoded)
    data_scaled = scaler.transform(data_imputed)

    Y_pred = model.predict(data_scaled)
    accuracy = accuracy_score(Y_test, Y_pred)

    print("Tasa de aciertos sin PCA: ", accuracy)
    print(f"Tiempo de ejecución sin PCA: ", time.time() - start_time)