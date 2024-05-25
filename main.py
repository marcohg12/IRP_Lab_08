from classifier import load_training_objs
from training import train_dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import pandas as pd
from classifier import get_model_accuracy
from plotter import plot_scatter, plot_biplot


def main():

    # Obtenemos el dataset
    dataset = fetch_openml(name = 'adult', version = 2)

    # Obtenemos las características del dataset y eliminamos el ID de las personas
    X = pd.DataFrame(dataset.data, columns = dataset.feature_names)
    X = X.drop(columns=['fnlwgt'])

    # Obtenemos las etiquetas del dataset
    y = pd.Series(dataset.target, name = 'income')

    # Particionamos el dataset para obtener un conjunto para entrenamiento y otro para pruebas
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    # Entrenamos el modelo y obtenemos el dataset reducido usado para el entrenamiento
    X_pca = train_dataset(X_train, y_train)

    #plot_scatter(X_pca, y, 'PC1', 'PC2', 'income', 'Reduced Dataset Using PCA')

    plot_biplot(X_train, y_train, X.columns, 'income', 'PC1', 'PC2', 'Loadings for PCA per Feature')

    # Cargamos los objetos en el módulo de clasificación
    load_training_objs()

    print(get_model_accuracy(X_test, y_test))

main()
