# --------------------------------------------------------------------------------------------
# Este archivo contiene funciones para desplegar gráficos para los datasets
# Puede desplegar gráficos de visualización de esparcimiento de datos por componentes y
# pesos de características por componente
# --------------------------------------------------------------------------------------------

import numpy as np
import plotly.express as px
import pandas as pd
import joblib
from constants import TRAINING_OBJS_DIR
import os

# Recibe el conjunto de características (ya tratato con PCA) y la variable destino para las características
# Recibe los nombres de las columnas de los ejes, y el nombre de la variable destino y el título
# Despliega en el navegador el gráfico de los datos
def plot_scatter(X_data_pca, target, xlabel, ylabel, hue, title):

    # Crea un dataframe para los componentes PCA y les asigna la variable destino
    dataframe = pd.DataFrame(data = X_data_pca, columns = [xlabel, ylabel])
    dataframe[hue] = target

    fig = px.scatter(dataframe, x = xlabel, y = ylabel, color = hue, title = title)
    fig.show()


# Recibe el conjunto de características (no tratado con PCA) y la variable destino para las características
# Recibe las columnas originales del dataset, el nombre de la variable destino, los nombres de los ejes y el título
# Genera un gráfico con los pesos de cada característica para las componentes
def plot_biplot(X_data, target, og_dataframe_columns, hue, xlabel, ylabel, title):

    # Crea un dataframe para las características y les asigna la variable destino
    dataframe = pd.DataFrame(data = X_data, columns = [xlabel, ylabel])
    dataframe[hue] = target

    # Carga el modelo PCA generado en el entrenamiento
    pca = joblib.load(os.path.join(TRAINING_OBJS_DIR, 'pca.pkl'))

    # Obtiene los pesos por característica y los carga en un dataframe
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)
    loadings_df = pd.DataFrame(loadings, columns = [xlabel, ylabel], index = og_dataframe_columns)

    fig = px.scatter(dataframe, x = xlabel, y = ylabel, color = hue, title = title, hover_data=[dataframe.index])
    
    # Genera las líneas para cada característica en el gráfico
    for feature in loadings_df.index:
        
        fig.add_scatter(x = [0, loadings_df.loc[feature, xlabel]], y = [0, loadings_df.loc[feature, ylabel]],
                        mode = 'lines', name = feature)
        
        fig.add_annotation(x = loadings_df.loc[feature, xlabel], y = loadings_df.loc[feature, ylabel],
                           text = feature, showarrow = True, ax = 0, ay = -40)
        
    fig.show()