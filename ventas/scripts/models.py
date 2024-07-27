from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# Cargar el archivo .parquet
file_path = 'C:/Users/Arley/Documents/Nueva carpeta/data/Ventas.parquet'  
df = pd.read_parquet(file_path)

# Limpiar los datos
df = df[df['Cantidad'] >= 0]

# Convertir la columna de fechas en columnas separadas
if 'Año/Mes' in df.columns:
    df[['Año', 'Mes']] = df['Año/Mes'].str.split('|', expand=True)
    df['Año'] = df['Año'].astype(int)
    df['Mes'] = df['Mes'].astype(int)
    df = df.drop('Año/Mes', axis=1)

# Convertir variables categóricas a numéricas
label_encoders = {}
for column in ['Centro', 'Material', 'Artículo', 'TALLA', 'COLOR']:
    if column in df.columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le


X = df.drop('Cantidad', axis=1)
y = df['Cantidad']
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

# Dividir los datos
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Regresión Lineal
pipeline_lr = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LinearRegression())
])
pipeline_lr.fit(X_train, y_train)
y_pred_lr = pipeline_lr.predict(X_test)

print(f'\nRegresión Lineal:')
print(f'Error Cuadrático Medio: {mean_squared_error(y_test, y_pred_lr):.2f}')
print(f'Error Absoluto Medio: {mean_absolute_error(y_test, y_pred_lr):.2f}')
print(f'Coeficiente de Determinación R²: {r2_score(y_test, y_pred_lr):.2f}')

# Random Forest
pipeline_rf = Pipeline([
    ('scaler', StandardScaler()),
    ('model', RandomForestRegressor(random_state=42))
])
param_grid_rf = {
    'model__n_estimators': [50, 100, 200],
    'model__max_depth': [None, 10, 20, 30]
}
grid_search_rf = GridSearchCV(pipeline_rf, param_grid_rf, cv=5, scoring='neg_mean_squared_error')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_
y_pred_rf = best_rf_model.predict(X_test)

print(f'\nRandom Forest (mejor ajuste):')
print(f'Error Cuadrático Medio: {mean_squared_error(y_test, y_pred_rf):.2f}')
print(f'Error Absoluto Medio: {mean_absolute_error(y_test, y_pred_rf):.2f}')
print(f'Coeficiente de Determinación R²: {r2_score(y_test, y_pred_rf):.2f}')

# Gradient Boosting
pipeline_gb = Pipeline([
    ('scaler', StandardScaler()),
    ('model', GradientBoostingRegressor(random_state=42))
])
param_grid_gb = {
    'model__n_estimators': [50, 100, 200],
    'model__learning_rate': [0.01, 0.1, 0.2],
    'model__max_depth': [3, 5, 7]
}
grid_search_gb = GridSearchCV(pipeline_gb, param_grid_gb, cv=5, scoring='neg_mean_squared_error')
grid_search_gb.fit(X_train, y_train)
best_gb_model = grid_search_gb.best_estimator_
y_pred_gb = best_gb_model.predict(X_test)

print(f'\nGradient Boosting (mejor ajuste):')
print(f'Error Cuadrático Medio: {mean_squared_error(y_test, y_pred_gb):.2f}')
print(f'Error Absoluto Medio: {mean_absolute_error(y_test, y_pred_gb):.2f}')
print(f'Coeficiente de Determinación R²: {r2_score(y_test, y_pred_gb):.2f}')
