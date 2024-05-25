# --------------------------------------------------------------------------------------------
# Este archivo contiene una función para entrenar un modelo supervisado usando componentes
# PCA, la función guarda los objetos usados en el entrenamiento
# --------------------------------------------------------------------------------------------

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
import numpy as np
import joblib
import os
from constants import TRAINING_OBJS_DIR

# Recibe el dataset y el nombre de la variable que se desea predecir
# Entrena un modelo utilizando PCA
# Para este caso se usan 2 PCA
# Retorna el dataset reducido utilizado para el entrenamiento
def train_dataset(X_train, Y_train):
    
    # Separamos las características de la variable que se desea predecir
    #X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    #y = pd.Series(dataset.target, name = target_name)
    
    # Codificamos las variables cualitativas
    ordinal_encoder = OrdinalEncoder(handle_unknown = 'use_encoded_value', unknown_value = np.nan)
    X_encoded = ordinal_encoder.fit_transform(X_train)

    # Reemplazamos datos faltantes por la media
    imputer = SimpleImputer(strategy = 'mean')
    X_imputed = imputer.fit_transform(X_encoded)
    
    # Estandarizamos los datos
    scaler = StandardScaler(with_mean=False)
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Ejecutamos un PCA para reducir la dimensionalidad a 2
    pca = PCA(n_components = 2)
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

    return X_pca