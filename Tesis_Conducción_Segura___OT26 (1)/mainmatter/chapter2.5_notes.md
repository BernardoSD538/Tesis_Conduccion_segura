# Cap 2

takeaways: 
- creo que en alguna parte del paper hay que mencionar lo del espectro de alerta del conductor
- pudiera jalar poner diagramas acá para lo del MAR/EAR
- Cambiar bases de datos por conjuntos de datos
- lo de sie s percision o exactitud
- cuidar que no estemos dando a entender que nosotros creemos que un algoritmo es capaz de decidir cosas, en el sentido de que como si un algoritmo fuera una persona. 

## 2.1 Detección de somnolencia

*Hay objetivas y subjetivas*

*Las subjetivas consisten en... entre las escalas usadas son... son faciles de usar*

> Tenemos la validez de estos métodos? Es necesaria? 

*debido a esto, es facil usar NNs, y se han creado muchos datasets; los mas populares son...*

*esta tabla muestra precisiones con escalas subjetivas. se usan modelos de ML/DL, se usan metricas vehiculo (???) y rostro*

- De nuevo, precisiones o exactitud?

*tabla*

- Se habla de "del estado del arte". Se me hace curiosa esa forma de expresarlo

*Por otro lado, las objetivas se obtienen... incluyendo fisiológicas y rostro. Las fis son invasivas, entonces usamos de rostro. Metricas comunes son...*

*Usualmente hay de dos: o pura VC, o incluir DL. la pura VC. Una..., la otra...*

- Creo que se pudiera hacer un mayor énfasis en el hecho de que las redes neuronales pueden ser más costly su inferencia, e incluso agregar referencia la de la de 20 o 30 segundos de inferencia. 

*la siguiente tabla muestra precisiones de los objetivos*

- btw, cuando se habla de "varianza" realmente hablamos de varianza o de variación?
- hmmm siento que estamos otra vez *repitiendo* el decir "fisiológicas como EEG, behavioral como PERCLOS..."

### 2.1.1 PERCLOS

*El perclos es... se puede medir con algoritmos integrados... pero pueden tener resultados limitados... por eso se usa procesamiento de imagenes... se usa dlib y opencv para capturar y procesar EAR... se usa EAR para PERCLOS*

- Hmm no sé si realmente todo el párrafo la referencia es Minhas.
- SmartEye no sé si sea un algoritmo
- Qué investigaciones los "algoritmos integrados" pueden "tener resultados limitados y no son tan confiables"? Referencias
- Corregir PECLOS

*raza se ha puesto a comprobar que exista relacion entre PERCLOS y somnolencia... han encontrado que existe alto nivel de precisión al comparar detecciones de somnolencia con métricas comprobadas y el PERCLOS, y correlación entre PERCLOS y métircas vehiculares*

- Sí estaría cool poner más referencias de esto. 

### 2.1.2 Duración de parpadeo

- 500ms va junto? 
- Hace falta incluir la referencia que habla del espectro de nivel de alerta de un conductor para decir no solo que lo del microsueño es la última, sino también la más peligrosa (o capaz es la de schleicher)

### 2.1.3 Bostezo

*el bostezo es indicador secundario porque... pero si comprueba que... por si solo no jala, pero se puede usar para decidir si alguien esta en somnolencia o ne... es parecido a lo del EAR: opencv -> dlib -> mar... en teoria va de 0-1, pero en realidad van de 0.14-0.9*

- Mencionar que sí es indicador de fatiga, pero no de somnolencia
- falta referencia para lo de los rangos del MAR

*es posible diferenciar MAR cuando alguien canta, habla, bosteza. se ve en el trabajo de [estos chavos]*

- estaria bien sí hacer la referencia más inline de los chavos que hicieron la fig 2.1

*Investigación de otros compas muestra que el threshold de 0.5 para el MAR era el equilibrio adecuado entre falsos negativos y falsos positivos*

### 2.1.4 Pose de cabeza

*es un indicador secuendario; pueden pasar muchos escenarios, pero a nos el unico que nos impoirta es el de cabeceo, que sucede cuando...*

- Falta la referencia de eso

*Existen algoritmos para la pose de la cabeza...*

- hmm no entiendo el por qué de este párrafo

> Aquí termina esta sección. SI MAL NO RECUERDO el chiste era esperarnos hasta la metodología para explicar cosas como:

**nosotros no vimos necesario el estimar la pose completa, con saber el pitch se arma... revisando literatura se infirió cuando se tiene una cara de frente, las landmarks 3D se proyectan a 2D de la siguiente forma cuando la cabeza gira en diferentes direcciones... con estas nociones en mente, y para aprovechar el tiempo con el que contabamos, se le encargó a un LLM diseñar el algoritmo de pitch... este es el algoritmo "refinado"...**

En cualquier caso, ndms habría que considerar esto:

- Ver si tenemos que incluir el prompt como tal o ndms hacer paráfrasis, porque tmb se tuvo que ajustar varias cosas
- lo de "refinado" es ndms en caso de que sí se haya refinado como tal; pdte revisitar converasción

## 2.2 Detección de distracción

*El uso de CNNs pa distracción esta bien documentado. Entre los datasets más comunes se encuentran...*

- No me convence simplemente ponerlas en viñetas. Creo estaría bien dejarlas in-line, a menos de que se quiera poner otra osa tipo imgs o asi

*la tabla muestra experimentacion... muestras muchas arquitecturas y resultados prometedores... las de ACU y state-farm se relacionanc on 99% de precisiones, solo una del DMD que obtuvo... lo cual sugiere que...*

- No queda claro por qué ese 73% sugiere arquitecturas complejas para obtener resultados de "precisiones" altas
- Neceistamos buscar mejores témrinos para describir lo que coloquialmente llamamos "simple" o "complejo"

*De acuerdo con el NHTSA... si el driver hace algo mas de 2 segs, entonces esta distraido... esta es una guia...*

*tabla 2.3*

## 2.3 Implementación en tiempo real

### 2.3.1 Threading

*el threading es... esto puedo mejorar la eficiencia porque... hay ventajas como cuando se necesita procesar mucha info...unos experimentaron con polinomios... sus resultados mostraorn que la distribución de tareas mejora el tiempo de ejecución... la figura muestra que...*

- realmente "hilos de ejecución" es la mejor traducción?
- creo que decir "algoritmos de tiempo real" no es exacto. 
- quizá decir "simultánea, es decir, al mismo tiempo"
- no se qué tanta distincion necesitemos hacer aquí entre paralelo, pseudo-paralelo (si es que eso existe), y concurrente. 
- hmmm creo que se puede cambiar la redaccion de lo de "procesar mucha ifnormacion". 
- pudieramos incluir algo acerca de como cabe mencionar que no es una relación estrictamente lineal la de el numero de threads vs el tiempo para el cual es terminaron de ejecutar todas las tareas. 

### 2.3.2 Redes en ONNX

**PENDIENTE**

### 2.3.3 Comparativa entre lenguajes Python y C++

*diferentes lenguajes pueden tener ventajas y desventajas... c++ se compila, py se interpreta...*

- quiza hubiera palabras mejores que "ventajas" y "desventajas"? pdte investigar
- "distintos lenguajes *de programación*"
- definitivamente al menos buscaría otro término para "interpretabilidad" porque no creo que el tema es que C++ sea menos interpretable en el sentido de que haya ambiguedades en lo que un determinado enunciado haga o no, sino que quizá la sintáxis es menos parecida al inglés y ya. PDTE
- no creo que sea adecuado "capacidades de compilación"
- "es decir, *el codigo fuente* se traduce..."
- agregar lo del preprocesador y el linking? 
- "el más rapido de *ambos*"
- compilación nativa?
- "*velocidades mayores en absolutamente*..."
- veo que la gráfica menciona lo de dataset size. pero el lector no tiene contexto de ningun dataset, y nosotros le estamos diciendo que c++ es mas rapido que python mientras hacemos referencia a ese dataset. Que implica ese dataset para el lector? cómo se traduce el hecho de que con los investigadores, a mayor dataset, mayor tiempo de ejecucion del algoritmo, a gente que no trabaja con ese dataset pero si con otros algoritmos y otras tareas? 