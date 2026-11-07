# Conclusiones del análisis exploratorio — Titanic

**Autor:** Samuel Acevedo Mambuscay
**Datos:** 891 pasajeros etiquetados (capa Silver)

Informe para la gerencia de la aseguradora sobre los factores asociados con la supervivencia en el naufragio del Titanic (1912).

---

## 1. Hallazgos

1. **El sexo es el factor más asociado con sobrevivir.** Sobrevivió el 74.2 % de las mujeres (n = 314) frente al 18.9 % de los hombres (n = 577). La prueba chi-cuadrado (χ² = 260.7, p ≈ 1.2 × 10⁻⁵⁸) indica que es prácticamente imposible que esta diferencia se deba al azar.

2. **La clase marca una gran diferencia.** La tasa de supervivencia cae con la clase: 63.0 % en 1.ª (n = 216), 47.3 % en 2.ª (n = 184) y 24.2 % en 3.ª (n = 491).

3. **Clase y sexo interactúan.** Las mujeres de 1.ª y 2.ª clase sobrevivieron casi todas (96.8 % y 92.1 %), mientras que las de 3.ª solo el 50.0 %. Entre los hombres la ventaja de la 1.ª clase también es clara: 36.9 % frente a 15.7 % en 2.ª y 13.5 % en 3.ª.

4. **Los niños sobrevivieron más, salvo en 3.ª clase.** Los menores de 16 años sobrevivieron en un 83.3 % en 1.ª clase (n = 6) y en un 100 % en 2.ª (n = 19), pero solo en un 43.1 % en 3.ª (n = 58). Ser niño protegía mucho menos si se viajaba en la clase más baja.

5. **Viajar con familia pequeña se asocia con más supervivencia.** Quienes viajaban con 1 a 3 familiares sobrevivieron en un 57.9 % (n = 292), frente al 30.4 % de quienes viajaban solos (n = 537) y al 16.1 % de las familias de 4 o más (n = 62).

6. **Pagar más se asocia con sobrevivir, pero solo en las clases altas.** Al comparar cuartiles de tarifa dentro de cada clase, la supervivencia sube del 46 % al 74 % en 1.ª clase y del 35 % al 59 % en 2.ª, mientras que en 3.ª clase se mantiene entre el 20 % y el 28 % sin importar la tarifa.

## 2. Decisiones de limpieza y su sesgo

- **Reto 8.1 — `Person` sin edad:** 177 pasajeros no tienen edad registrada y, por la regla `Age < 16`, ninguno puede clasificarse como `child`. Si tuvieran la misma proporción de niños que los pasajeros con edad conocida (11.6 %), faltarían **≈ 21 niños**: la variable `Person` subestima a los niños.
- **Reto 22.1 — edad imputada:** al rellenar la edad con el promedio por sexo, la desviación estándar baja de **14.53 a 13.01** años, porque se añaden 177 valores idénticos. En el histograma aparecen dos picos artificiales en **27.9 años** (mujeres) y **30.7 años** (hombres). Por eso Silver conserva `AgeOriginal`, y el modelo imputa solo con datos de entrenamiento.
- **`Embarked`:** 2 valores faltantes, rellenados con la moda (`S`). Su efecto en el análisis es despreciable.
- **`X` en cabina:** el **77.1 %** de los pasajeros no tiene cabina registrada. Se conservó como categoría "desconocida" en lugar de eliminar esas filas, aunque mezcla dos situaciones distintas: no tenía cabina o no se registró.
- **`Master` → `Mr`:** la consolidación de títulos une a los niños varones con los hombres adultos. Es defendible para tener grupos con tamaño suficiente, porque la variable `Person` conserva la distinción entre niños y adultos.

## 3. Confusiones detectadas

- **Puerto vs. clase:** a primera vista, los pasajeros embarcados en Cherbourg (C) sobrevivieron más (55.4 %, n = 168) que los de Queenstown (Q, 39.0 %, n = 77) y Southampton (S, 33.9 %, n = 646). Pero Cherbourg trajo una proporción mucho mayor de pasajeros de 1.ª clase (50.6 %, frente al 20.0 % de S y el 2.6 % de Q), mientras que el 93.5 % de los embarcados en Queenstown viajaba en 3.ª clase. Al **estratificar por clase**, la diferencia se reduce: dentro de la 1.ª clase, C tiene 69 % frente a 59 % de S. Que no desaparezca del todo (en 3.ª clase, C y Q tienen 38 % frente a 19 % de S) sugiere que otras variables, como la composición por sexo de cada puerto, explican parte del resto. **No hay evidencia de que el puerto en sí influyera en la supervivencia.**
- **Tarifa vs. clase:** la tarifa está muy ligada a la clase, así que compararla directamente confundiría ambos efectos. Por eso se analizó por cuartiles **dentro de cada clase**: la tarifa solo aporta información adicional en 1.ª y 2.ª clase.

## 4. Qué NO podemos afirmar

- **Asociación no es causalidad.** Que las mujeres y los niños sobrevivieran más refleja la regla de evacuación "mujeres y niños primero" y el acceso a los botes según la ubicación en el barco, no una característica del sexo o la edad en sí.
- **Un solo naufragio.** Los patrones de 1912 no se pueden generalizar a otros desastres, barcos o protocolos actuales.
- **Datos incompletos.** Falta el 19.9 % de las edades y el 77.1 % de las cabinas; las conclusiones sobre esas variables dependen de cómo se trataron los faltantes.
- **Grupos pequeños.** Las tasas basadas en pocos pasajeros (niños de 1.ª clase con n = 6, familias numerosas, títulos poco frecuentes) no son confiables.