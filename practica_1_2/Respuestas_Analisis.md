# Práctica 2 (Parcial 2) — Deep Learning: Detección de fallas en equipos de refrigeración
## Respuestas a las actividades de interpretación y análisis

> **Notebook:** `Practica_DeepLearning_Refrigeracion.ipynb` (ejecutado, con gráficas y resultados incrustados)
> **Asignatura:** Inteligencia Artificial · **Tema:** Aprendizaje profundo · **Biblioteca:** TensorFlow / Keras

---

## 1. Resultados obtenidos (evidencia de la ejecución)

| Elemento | Resultado |
|---|---|
| Registros generados | 2,800 (700 por condición: NORMAL, COMPRESOR, FUGA, PUERTA) |
| División de datos | 2,240 de entrenamiento (80%) / 560 de prueba (20%) |
| Arquitectura de la red | Entrada(5) → Dense 32 ReLU + Dropout 0.15 → Dense 16 ReLU → Salida 4 Softmax |
| Épocas utilizadas | 50 (EarlyStopping con paciencia 6, sin sobreajuste visible) |
| **Precisión del modelo** | **99.46 %** (pérdida: 0.0063) |
| Reporte de clasificación | COMPRESOR 1.00 · FUGA 0.99 · NORMAL 0.99 · PUERTA 1.00 (accuracy 0.99) |
| Prueba A (equipo normal) | Diagnóstico: **NORMAL** — confianza 100.00 % ✔ |
| Prueba B (compresor) | Diagnóstico: **COMPRESOR** — confianza 100.00 % ✔ |
| Prueba C (fuga) | Diagnóstico: **FUGA** — confianza 100.00 % ✔ |
| Prueba D (puerta) | Diagnóstico: **PUERTA** — confianza 100.00 % ✔ |

**Gráficas generadas en el notebook:**
1. Temperatura vs. vibración por condición (dispersión).
2. Precisión y pérdida durante el entrenamiento (entrenamiento vs. validación).
3. Matriz de confusión de las cuatro condiciones.
4. Panel de control con las probabilidades del diagnóstico.

---

## 2. Actividad de interpretación (Paso 6)

**1. ¿Qué condición presenta temperaturas más elevadas?**
La condición **PUERTA** (problema de puerta o aislamiento), con una temperatura media cercana a los **-10 °C**, es decir, la más alejada de los -20 °C de operación normal. Le sigue COMPRESOR (≈ -12 °C).

**2. ¿Qué condición presenta mayor vibración?**
La condición **COMPRESOR**, con una vibración media de aproximadamente **6 mm/s** (frente a 2–2.8 mm/s de las demás condiciones), ya que el daño en el compresor provoca vibración excesiva.

**3. ¿Es posible distinguir las fallas utilizando únicamente una variable?**
**No de forma confiable.** Cada variable por sí sola presenta zonas de solapamiento: por ejemplo, la temperatura de NORMAL, FUGA y COMPRESOR se llega a cruzar; por sí sola la vibración no distingue NORMAL de FUGA o PUERTA. Solo la combinación de todas las variables permite separar las cuatro condiciones.

**4. ¿Por qué conviene utilizar varias variables simultáneamente?**
Porque cada falla se manifiesta en una **combinación distinta** de parámetros (la fuga se nota en la presión baja, el compresor en vibración y corriente altas, la puerta en la apertura prolongada). Al usar las cinco variables a la vez, la red neuronal encuentra relaciones entre ellas y construye fronteras de decisión más claras, lo que aumenta la precisión y reduce los diagnósticos erróneos.

---

## 3. Actividad de análisis (Sección 16) — 10 preguntas

**1. ¿Qué es Deep Learning?**
Es un subconjunto del aprendizaje automático que utiliza **redes neuronales artificiales con varias capas ocultas** (aprendizaje profundo) para aprender automáticamente representaciones jerárquicas de los datos, sin que el programador tenga que definir a mano las características relevantes. En esta práctica, la red aprendió a reconocer patrones de falla a partir de 5 variables de entrada.

**2. ¿Cuál es la diferencia entre Machine Learning y Deep Learning?**
En el **Machine Learning** tradicional el humano debe hacer la *ingeniería de características* (seleccionar y preparar los datos) y luego alimentar un algoritmo (árbol de decisiones, SVM, etc.). En el **Deep Learning** la propia red, mediante sus múltiples capas de neuronas, **aprende las características** de los datos de forma automática. A cambio, el Deep Learning requiere más datos y mayor poder de cómputo para entrenarse.

**3. ¿Por qué se utilizaron cinco variables de entrada?**
Porque son los cinco parámetros que describen el funcionamiento del equipo (temperatura, vibración, corriente, presión y apertura de puerta), cada uno aporta información distinta de la falla y además la **capa de entrada tiene una neurona por cada variable** (Input(shape=(5,))). Con más variables la red puede detectar combinaciones anormales que con una sola variable pasarían desapercibidas.

**4. ¿Qué función cumplen las capas ocultas?**
Reciben la información de la capa anterior, la procesan mediante operaciones matemáticas (suma ponderada + función de activación ReLU) y transmiten el resultado a la siguiente capa. Permiten **extraer patrones y relaciones no lineales** entre las variables (por ejemplo: "presión baja + temperatura alta ⇒ posible fuga"). Las dos capas ocultas (32 y 16 neuronas) son las que hacen posible el aprendizaje profundo.

**5. ¿Qué sucedería si se incrementara el número de neuronas?**
La red tendría **mayor capacidad de aprendizaje** para representar patrones más complejos y podría mejorar la precisión si el problema es difícil. Sin embargo, también aumentan el tiempo de entrenamiento y el riesgo de **sobreajuste** (memorizar los datos de entrenamiento sin generalizar), por lo que se tendría que compensar con más datos o mayor regularización (Dropout).

**6. ¿Por qué se dividieron los datos en entrenamiento y prueba?**
Para **evaluar la capacidad de generalización** del modelo. La red solo aprende con el 80% de entrenamiento; al probarla con el 20% de datos que **nunca vio**, se comprueba si realmente aprendió los patrones y no simplemente memorizó los registros. Sin esta división, la precisión medida sería engañosa.

**7. ¿Qué función tiene el algoritmo Adam?**
Es el **optimizador**: ajusta (actualiza) los pesos de las conexiones de la red para minimizar la función de pérdida. Combina descenso de gradiente con momento y tasa de aprendizaje adaptativa por parámetro, por lo que converge de forma rápida y eficiente durante el entrenamiento.

**8. ¿Por qué se utiliza Softmax en la última capa?**
Porque es un problema de **clasificación multiclase** (4 condiciones). Softmax convierte las cuatro salidas de la red en **probabilidades que suman 100%**, permitiendo seleccionar la categoría con mayor valor (`np.argmax`) y reportar el porcentaje de confianza del diagnóstico.

**9. ¿Qué significa que el modelo tenga una precisión del 95%?**
Significa que el **95% de los registros del conjunto de prueba fueron clasificados correctamente**, es decir, 5 de cada 100 equipos serían diagnosticados de forma incorrecta. (En esta práctica se obtuvo **99.46%**.)

**10. ¿Qué consecuencias tendría que el modelo clasificara un equipo defectuoso como normal?**
Sería un **falso negativo**, el error más grave del sistema: la falla pasaría desapercibida y el equipo seguiría operando, provocando pérdida de la temperatura de conservación, deterioro de los alimentos congelados, mayor consumo eléctrico, daños al compresor, paradas no planificadas y costos de mantenimiento correctivo; en una empresa real también implicaría pérdidas económicas y posibles riesgos sanitarios.

---

## 4. Conclusión: utilidad de Deep Learning en el mantenimiento de equipos de refrigeración

Deep Learning permite transformar datos de operación (temperatura, vibración, corriente, presión y apertura de puerta) en **diagnósticos oportunos** de las condiciones de los equipos de refrigeración. En lugar de depender únicamente de inspecciones manuales, la red neuronal aprende los patrones asociados a cada falla y avisa de forma inmediata si una condición es anormal, lo que ayuda a evitar la pérdida de alimentos congelados, reducir el consumo eléctrico, prevenir daños en los compresores y disminuir los costos de mantenimiento correctivo.

Con solo dos capas ocultas, la red logró una precisión del **99.46%** y diagnosticó correctamente las cuatro condiciones probadas, demostrando que es posible crear un prototipo de mantenimiento predictivo sin comprar sensores ni instalar infraestructura, utilizando datos simulados. Aunque este modelo es educativo, en una instalación real los síntomas pueden coincidir entre fallas, por lo que el diagnóstico debe validarse con mediciones, documentación técnica y personal especializado.

---

## 5. Lista de verificación del video (3 minutos)

- [ ] Notebook `Practica_DeepLearning_Refrigeracion.ipynb` abierto en Google Colab.
- [ ] Código completo visible y ejecutable (celdas en orden).
- [ ] Conjunto de datos generado (2,800 registros, `df.head()` y `value_counts()`).
- [ ] Gráfica de entrenamiento y validación (precisión y pérdida).
- [ ] Matriz de confusión + reporte de clasificación.
- [ ] Las cuatro pruebas de diagnóstico (A, B, C, D) con su confianza.
- [ ] Panel de control (gráfica de barras del diagnóstico).
- [ ] Conclusión sobre la utilidad de Deep Learning en el mantenimiento.
