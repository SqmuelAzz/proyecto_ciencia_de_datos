# Conclusiones del EDA — Titanic

## 1. Hallazgos
1. **El sexo es el factor más asociado con sobrevivir.** Sobrevivió el ___ % de las mujeres (n = ___) frente al ___ % de los hombres (n = ___). Chi-cuadrado: p ≈ ___.
2. **La clase marca una gran diferencia.** 1.ª ___ % (n = ___), 2.ª ___ % (n = ___), 3.ª ___ % (n = ___).
3. **Clase y sexo interactúan.** Las mujeres de 1.ª y 2.ª clase sobrevivieron en más del ___ %; las de 3.ª solo el ___ %.
4. **Los niños sobrevivieron más que los hombres adultos** en todas las clases, pero en 3.ª clase su tasa fue baja (___ %, n = ___).
5. **Viajar con 1 a 3 familiares** se asocia con mayor supervivencia (___ %) que viajar solo (___ %, n = ___).

## 2. Decisiones de limpieza y su sesgo
- **Reto 8.1 — `Person` sin edad:** ___ pasajeros sin edad nunca pueden ser `child`; faltarían ≈ ___ niños.
- **Reto 22.1 — edad imputada:** la desviación estándar baja de ___ a ___ y aparecen picos artificiales en ___ y ___ años.
- **`Embarked`:** ___ faltantes rellenados con `S`; efecto despreciable.
- **`X` en cabina:** el ___ % sin cabina; se conserva como categoría "desconocida".
- **`Master` → `Mr`:** mezcla niños con adultos; `Person` conserva la distinción.

## 3. Confusiones detectadas
- **Puerto vs. clase:** C parece tener más supervivencia, pero trajo más pasajeros de 1.ª clase; al estratificar por clase la diferencia ___.
- **Tarifa vs. clase:** se comparó por cuartiles dentro de cada clase: ___.

## 4. Qué NO podemos afirmar
- Asociación ≠ causalidad (las tasas reflejan la regla "mujeres y niños primero").
- Es un solo naufragio: no se generaliza sin más a otros desastres.
- Falta el ___ % de las edades y el ___ % de las cabinas.
- Las tasas con n pequeño (familias grandes, títulos raros, cubierta T) no son confiables.