# Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Cargar el archivo .parquet
file_path = 'data/Ventas.parquet' 
df = pd.read_parquet(file_path)

# Eliminar valores negativos en la columna 'Cantidad'
df = df[df['Cantidad'] >= 0]

# Mostrar las primeras filas del DataFrame
print("Primeras filas del DataFrame:")
print(df.head())

# Información general del DataFrame
print("\nInformación del DataFrame:")
print(df.info())

# Estadísticas descriptivas
print("\nEstadísticas descriptivas:")
print(df.describe())

# Análisis exploratorio de datos
# Histograma de la cantidad
plt.figure(figsize=(10, 6))
sns.histplot(df['Cantidad'], kde=True)
plt.title('Distribución de Cantidades')
plt.xlabel('Cantidad')
plt.ylabel('Frecuencia')
plt.show()

# Correlaciones

df_encoded = pd.get_dummies(df, columns=['Centro', 'Material', 'Artículo', 'TALLA', 'COLOR'])

# Calcular la matriz de correlación
correlations = df_encoded.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlations, annot=False, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# Preprocesamiento de datos
X = df_encoded.drop('Cantidad', axis=1)
y = df_encoded['Cantidad']

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar un modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'\nError Cuadrático Medio: {mse:.2f}')
print(f'Error Absoluto Medio: {mae:.2f}')
print(f'Coeficiente de Determinación R²: {r2:.2f}')

# Importancia de las características
importances = model.coef_
features = X.columns
feature_importances = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=feature_importances.index)
plt.title('Importancia de las Características')
plt.xlabel('Importancia')
plt.ylabel('Características')
plt.show()