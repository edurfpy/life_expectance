# Life_expectance
Predicción de la esperanza de vida. 
Datos de múltiples países relativos a factores económicos (país desarrollado o en vías, producto interior bruto), de salud (mortalidad, tasa de obesidad, incidencia distintas enfermedades) y sociales (tasa escolarización, ingresos)
Carga de datos, preprocesado y prueba distintos modelos para predicción de la esperanza de vida


## PREPROCESADO: Carpeta src > Custom_preprox
Primeras acciones de limpieza usualmente presentes:
   - eliminar registros (filas) duplicados
   - eliminar registros con nulos en el target
   - eliminar atributos (columnas) con varianza cercana a cero
   - eliminar registros con nulos a partir de un nº, estableciendo threshold de valores válidos.

Sustituir valores no coherentes en cada variable si el nº de valores es menor a un umbral especificado (variables tipo porcentaje, en un rango max-min o estrictamente positivas) por un valor : class AssignWrongValuesTransformer
Si el nº es mayor opción degún parámetro de eliminar la variable

Transformador particularizado con aplicación de Power Transformer opcional: class CustomLexpectPreprx


## ENTRENAMIENTO DISTINTOS MODELOS: archivo src > train.py
Training de una serie de modelos especificados, con un modelo base referencia para los resultados.

Presentación de resultados distintos modelos.

Almacenado de los n (parámetro) primeros mejores modelos 


## SEGUIMIENTO DEL PROCESO.
Creación de archivos log (carpeta src > log)


## ALMACENAMIENTO MODELOS ENTRENADOS.
Modelos entrenados (.pkl) y comparativa resultados (carpeta models)


## CARPETA OUTPUT.
Gráficas resultado EDA


## CARPETA UTILS.
Distintas utilidades necesarias (EDA, etc)


Ejecutando “run_training.py” iniciamos la aplicación (carga, preprocesado, entrenamiento)
