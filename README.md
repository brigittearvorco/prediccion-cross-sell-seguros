# Predicción de Interés en Cross-Sell de Seguro Vehicular

Modelo de regresión logística que predice qué clientes de seguro de salud
tienen mayor probabilidad de estar interesados en adquirir un seguro
vehicular, a partir de variables demográficas, de póliza y de vehículo.

## Dataset
Health Insurance Cross Sell Prediction (Kaggle) — 381,109 registros,
11 variables predictoras. Variable objetivo (`Response`) desbalanceada:
87.7% no interesados, 12.3% interesados.

## Proceso
1. Análisis Exploratorio de Datos (EDA) para confirmar el desbalance de clases.
2. Codificación de variables categóricas (Gender, Vehicle_Age, Vehicle_Damage) con one-hot encoding.
3. Escalado de variables numéricas con StandardScaler.
4. Entrenamiento de un modelo de regresión logística con `class_weight='balanced'`
   para compensar el desbalance de clases — sin este ajuste, el modelo no
   detectaba ningún cliente interesado (recall de 0.00 en la clase minoritaria).
5. Evaluación con matriz de confusión, precisión/recall y AUC.

## Resultados
- AUC: 0.84
- Recall (clase "Interesado"): 0.97 — el modelo identifica casi a todos los
  clientes realmente interesados.
- Precisión (clase "Interesado"): 0.25 — de cada 4 clientes marcados como
  interesados, 1 lo es realmente.

## Interpretación de negocio
El modelo prioriza recall sobre precisión: para una campaña de cross-sell,
el costo de contactar a alguien no interesado es menor que el costo de
no identificar a alguien que sí compraría. El umbral de decisión podría
ajustarse según el costo real de contacto vs. el valor esperado de una venta.

## Herramientas
Python, pandas, scikit-learn, seaborn, matplotlib
