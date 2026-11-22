# Cap 3

## 3.1 Detección de somnolencia

### 3.1.1 Detección de rostro

*necesitamos detectar primero rostros. para eso hay varias opciones, entre ellas el HOG. el HOG es...*

- Es una descripció ligera, entonces tmp hace daño que esté aquí, solo quería recordar que no es *necesario* que esté descrito el HOG. 

### 3.1.2 Cálculo de PERCLOS y duración de parpadeo

**PENDIENTE**

### 3.1.3 Detección de bostezo

**PENDIENTE**

### 3.1.4 Cálculo de pose de cabeza

**PENDIENTE**

## 3.2 Detección de distracción

### 3.2.1 Características de conjunto de datos

*el dataset es el DMD. Fue recolectado por... es multimodal... los participantes... hay sims de carros en mov, estaticos, y en simuladores... cada sesion cuenta con... la fig muestra ejemplos... las imgs son RGB... tiene varias anotaciones...*

*nosotros usamos solo las de mirada para claisif binaria... se muestran ejemplos...*

*figuras*

*necesitamos hacer un balance de clases en el entrenamiento...*

### 3.2.2 Modelos de aprendizaje profundo entrenados

*se entrenaron varias arquitecturas para encontrar el sweet spot entre accuracy y eficiencia computacional... empezamos con sencillas luego con mas complejas o con mas params... se optimizaron hiperparams con Optuna...se entrenó una vez la LSTM luego se optimizaron hiperparams*

*la tabla 3.1 muestral os experimentos...*

### 3.2.2 Modelos de aprendizaje profundo entrenados

*con las CNNs entrenadas, se implementaron en tiempo real. la LSTM se le aplicó una ventana de 10 frames de entrada, y necesitan haber predicciones positivas durante 2 segundos para que se arroje una alarma de distracción*

*tabla 3.1*

*se realizaron pruebas de ese código para probarlo en condiciones optimas...*

*tmb en una RPi 5B... specs...*

## 3.3 Especificaciones de equipo y algoritmo en tiempo real

## 3.3.1 Equipo *Plug-and-play*

**PENDIENTE**

## 3.3.2 Modularización de algoritmo y cambio de lenguaje Python a C++

**PENDIENTE**