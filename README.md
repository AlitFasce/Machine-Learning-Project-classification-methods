# Machine-Learning-Project-classification-methods

_ PROYECTO INDIVIDUAL Nº2 para Henry_ Diciembre 2022 _ Machine Learning

Este es un proyecto de Machine Learning para predecir si una estadía de un paciente en un hospital determinado será larga o corta, utilizando la información disponible en datasets. Se aplican varios modelos y se compara la performance de estos. Como es un proyecto de aprendizaje les comparto la mayor cantidad de información posible, estrategia original y final; y porque tomé las distintas decisiones. 

![imagen](https://metasdigitais.com.br/wp-content/uploads/2020/04/marketing-digital-para-pequenas-e-medias-empresas.jpg)


## **Introducción**

Soy Alit Fasce, en esta oportunidad estoy realizando este proyecto, como parte de los laboratorios de Henry el mismo tiene una orientación sobre Machine learning sin descuidar temas de EDA y transformación de datos que son siempre tan importantes para lograr un buen modelo. 

Puedes ver el [Programa Data Science - Henry](https://www.soyhenry.com/carrera-data-science)

## **Descripción del problema**

Un Centro de Salud tiene por objetivo de poder predecir si un paciente tendrá una estancia hospitalaria prolongada o no, utilizando la información contenida en el dataset provisto, la cual recaba una muestra histórica de sus pacientes, para poder administrar la demanda de camas en el hospital según la condición de los pacientes recientemente ingresados.

Para esto, se define que un paciente posee estancia hospitalaria prolongada si ha estado hospitalizado más de 8 días. Por lo que debe generar dicha variable categórica y luego categorizar los pacientes según las variables que usted considere necesarias, justificando dicha elección.​
[Consigna completa del PI](https://github.com/soyHenry/Datathon)

## *Trabajo realizado y Criterio**

- El objetivo de este trabajo es lograr la estimación de un problema de clasificación mediante modelos de Machine Learning (ML). 

- Mi abordaje personal tiene dos puntos importantes:
    - el primero, de aprendizaje, basado en el armado de una estratégia incial y como esta fue mutando debido al descubrimiento de herramientas o mejores criterios de gestion de los datos.
    - la segunda, más técnica, enfocada en comenzar analizando el dataset, luego transformar los datos, analizar la correlación e implementar 3 modelos de ML que permitan obtener predicciones, finalmente elegir entre el mejor* de estos (se presenta la estrategia incial y una final como archivos txt).

- Es importante tener en cuenta que se trabajó en Colab, ya que muchos de los procesos de entrenamiento consumian gran capacidad de computo. 

- Finalmente, se realizó un intento de implementar una red neuronal con 2 capas en Tensorflow (tambien realizado en Colab)

*considero mejor, aquel dataset que balancea las metricas y por ende los niveles de falsos positivos y falsos negativos, de la matriz de confusión, intentado que ambos se mantengan relativamente bajos. 

- SE UTILIZA NOTION para gestion del projecto [ver tablero de gestion de proyecto](https://www.notion.so/e90a08753ab344d19bf21e934f5646a4?v=cdeed599f61749c49aaf4385540363c3)

## Pasos realizados 

- Análisis exploratorio de los datos (EDA).

- Transformación de datos

- Analisis de correlacion 

- Eleccion de los features

- División de dataset en train y test utilizando train_test_split, y similares.

- Elección de los modelos de clasificación

- Entrenamiento, optimizacion del hiperparametros y validación cruzada cuando se requirió

- Revisión de features y reentrenamiento cuando se requirió

- Aplicación de Pipeline al proceso


## **Estructura de Carpetas**


- EDA_y_Transformación
    - archivo jupiter Notebook (con anotaciones en markdown de los pasos realizados) 
    - EDA Report txt

    - Carpeta: Datasets
        - hospitalizaciones_test.csv (archivo utilizado unicamente para las predicciones)
        - hospitalizaciones_train.csv (archivo raw de trabajo)


- Logistica
    - archivo jupiter Notebook (con anotaciones en markdown de los pasos realizados)
	
    - Carpeta: csv_predicciones - IN PROGRESS
		- alitfasce.csv (archivo generado luego de estimación por logistica)


- K-vecions
    - archivo jupiter Notebook (con anotaciones en markdown de los pasos realizados)
	
    - Carpeta: csv_predicciones
		- alitfasce.csv (archivo generado luego de estimación por k-vecinos)
    - Modelos entrenados (3 differentes)


- Arbol_de_decision
    - archivo jupiter Notebook (con anotaciones en markdown de los pasos realizados)
	
    - Carpeta: csv_predicciones - IN PROGRESS
		- alitfasce.csv (archivo generado luego de estimación por Arbol_de_decision)

- Red_Tensorflow
    - archivo jupiter Notebook (con anotaciones en markdown de los pasos realizados)
	
    - Carpeta: csv_predicciones - IN PROGRESS
		- alitfasce.csv (archivo generado luego de estimación por DNN)


 Archivos globales: 

-	README

## **Herramientas utilizadas y documentacion**

    * Python - https://docs.python.org/3/

    * Sklearn -  https://scikit-learn.org/stable/

    * Pandas -  https://pandas.pydata.org/docs/
    
    * Numpy -  https://numpy.org/doc/
    
    * Matplotlib - https://matplotlib.org/stable/index.html
    
    * Seaborn - https://seaborn.pydata.org/
    
    * Pipeline - https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html

    * Joblib - https://joblib.readthedocs.io/en/latest/

    * Pathlib - https://pathlib.readthedocs.io/en/pep428/

    *Tensorflow https://www.tensorflow.org/guide

    *DNN clasifier https://www.tensorflow.org/api_docs/python/tf/estimator/DNNClassifier

    *Auto Gluon https://auto.gluon.ai/stable/index.html

