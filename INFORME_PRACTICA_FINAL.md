# INFORME FINAL DE PRÁCTICA
## Optimización Heurística del Problema del Agente Viajero (TSP) y Ruteo de Vehículos con Capacidad (CVRP)
### Caso de Estudio: Reparto Artesanal de Pan a Tiendas de Abarrotes en Querétaro

---

**Materia:** Inteligencia Artificial / Cómputo Evolutivo  
**Proyecto:** Algoritmos Genéticos Implementados desde Cero  
**Fecha:** Noviembre 2026  
**Lenguaje:** Python 3.12 / 3.14 (Librerías estándar: NumPy, Matplotlib, Pandas; sin DEAP ni PyGAD)  

---

## 1. Introducción y Planteamiento del Problema

### 1.1 Contexto Operativo
Una panificadora artesanal ubicada en el municipio de Querétaro reparte pan diariamente a **20 tiendas de abarrotes** distribuidas en la zona metropolitana. La logística de distribución parte desde el depósito central (**D**), visita cada una de las 20 tiendas exactamente una vez y concluye retornando a **D**.

El problema central se modela como un **Problema del Agente Viajero (TSP)** simétrico en $\mathbb{R}^2$. Adicionalmente, se resuelve la extensión al **Problema de Ruteo de Vehículos con Capacidad (CVRP)**, considerando que la camioneta de reparto tiene una capacidad máxima de **100 kg** y la demanda acumulada de las tiendas asciende a **270 kg**, lo que impone una cota inferior teórica de:
$$\text{Camionetas}_{\min} = \left\lceil \frac{270\text{ kg}}{100\text{ kg}} \right\rceil = 3\text{ viajes}$$

### 1.2 Datos del Problema (`puntos.csv`)
Las coordenadas en kilómetros y las demandas en kilogramos se resumen a continuación:

| ID | Tipo de Nodo | Coordenada X (km) | Coordenada Y (km) | Demanda (kg) |
| :---: | :---: | :---: | :---: | :---: |
| **D** | Depósito Central | 10.0 | 10.0 | 0 |
| **P01** | Tienda 01 | 4.2 | 14.8 | 15 |
| **P02** | Tienda 02 | 16.5 | 15.3 | 10 |
| **P03** | Tienda 03 | 8.1 | 3.6 | 12 |
| **P04** | Tienda 04 | 18.4 | 7.2 | 18 |
| **P05** | Tienda 05 | 1.5 | 9.4 | 8 |
| **P06** | Tienda 06 | 12.7 | 18.1 | 20 |
| **P07** | Tienda 07 | 6.8 | 11.9 | 14 |
| **P08** | Tienda 08 | 14.3 | 11.2 | 9 |
| **P09** | Tienda 09 | 3.3 | 18.0 | 16 |
| **P10** | Tienda 10 | 17.9 | 1.8 | 11 |
| **P11** | Tienda 11 | 9.5 | 16.4 | 13 |
| **P12** | Tienda 12 | 2.6 | 4.1 | 22 |
| **P13** | Tienda 13 | 13.8 | 5.5 | 7 |
| **P14** | Tienda 14 | 7.4 | 7.1 | 17 |
| **P15** | Tienda 15 | 19.2 | 12.9 | 10 |
| **P16** | Tienda 16 | 5.5 | 1.2 | 19 |
| **P17** | Tienda 17 | 11.6 | 13.4 | 12 |
| **P18** | Tienda 18 | 1.2 | 14.1 | 15 |
| **P19** | Tienda 19 | 15.8 | 18.9 | 8 |
| **P20** | Tienda 20 | 10.9 | 1.0 | 14 |

La matriz de distancias euclidianas $21 \times 21$ se calcula vectorialmente:
$$D_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$$

---

## 2. Diseño del Algoritmo Genético

### 2.1 Representación del Cromosoma
Para evitar redundancia y fijar el inicio y fin en el depósito, el cromosoma se codifica como una **permutación pura de enteros** de longitud 20:
$$\vec{c} = [c_1, c_2, \dots, c_{20}], \quad c_i \in \{1, 2, \dots, 20\}, \quad c_i \ne c_j \ \forall i \ne j$$
El depósito $D$ (índice 0) se agrega implícitamente al evaluar la ruta: $0 \to c_1 \to c_2 \to \dots \to c_{20} \to 0$.

### 2.2 Validación de Factibilidad
Se utiliza una aserción estricta después de cada cruza y mutación:
```python
def es_valido(cromo, n=20):
    return sorted(cromo) == list(range(1, n + 1))
```

---

## 3. Respuestas a Preguntas de la Rúbrica

### 3.1 Pregunta 7: Funciones de Aptitud y sus Diferencias Teóricas
El TSP es un problema de minimización de distancia $L$, mientras que los Algoritmos Genéticos operan mediante maximización de aptitud $f$. Se diseñaron e implementaron tres formulaciones:

#### 1. Inversa Hiperbólica:
$$f_1(\vec{c}) = \frac{1}{L(\vec{c})}$$
- **Propiedad**: No lineal. Conforme $L$ disminuye, la pendiente $\left|\frac{df_1}{dL}\right| = \frac{1}{L^2}$ aumenta.
- **Efecto evolutivo**: Magnifica las diferencias entre cromosomas de alta calidad en las generaciones finales, incrementando la presión selectiva sobre los mejores individuos. Es estrictamente positiva ($f_1 > 0$), haciéndola directamente compatible con la ruleta de selección.

#### 2. Lineal Complementaria:
$$f_2(\vec{c}) = \max(0, L_{\max} - L(\vec{c}))$$
- **Propiedad**: Lineal. Cualquier decremento de distancia $\Delta L$ otorga exactamente el mismo beneficio $\Delta f_2$, independientemente de si la ruta base es buena o mala.
- **Efecto evolutivo**: Mantiene una presión de selección homogénea a lo largo de toda la corrida, previniendo la dominancia prematura de superindividuos en las primeras generaciones.

#### 3. Negativa Directa:
$$f_3(\vec{c}) = -L(\vec{c})$$
- **Propiedad**: Equivalencia matemática exacta de minimización. Adecuada para selección por torneo o ranking lineal donde solo se evalúa la relación de orden ($f_A > f_B$).

---

### 3.2 Pregunta 10: Demostración Manual de Cruza de 8 Genes (Corte [3:6])
Para evaluar operadores de permutación sin casos triviales, se seleccionó un corte multigen de **3 genes** (posiciones 3 a 5, slice `[3:6]`):
- **Padre 1 ($P_1$)**: `[1, 2, 3, | 4, 5, 6 |, 7, 8]`
- **Padre 2 ($P_2$)**: `[8, 5, 2, | 1, 3, 6 |, 4, 7]`

#### A. Order Crossover (OX) - Paso a Paso
1. **Copiar segmento central**:
   $$\text{Hijo}_1 = [\_, \_, \_, \mathbf{4, 5, 6}, \_, \_]$$
   Conjunto de genes ya asignados: $\{4, 5, 6\}$.
2. **Secuencia circular de $P_2$ desde índice $b = 6$**:
   $$\text{Índices}: 6, 7, 0, 1, 2, 3, 4, 5 \implies [4, 7, 8, 5, 2, 1, 3, 6]$$
3. **Filtrar genes ya presentes**:
   - Se descartan 4, 5 y 6.
   - Genes disponibles en orden circular: `[7, 8, 2, 1, 3]`.
4. **Llenar huecos circularmente a partir de la posición 6**:
   - Posición 6 $\to$ `7`
   - Posición 7 $\to$ `8`
   - Posición 0 $\to$ `2`
   - Posición 1 $\to$ `1`
   - Posición 2 $\to$ `3`
5. **Resultados OX**:
   $$\mathbf{\text{Hijo}_1^{\text{OX}} = [2, 1, 3, 4, 5, 6, 7, 8]}$$
   $$\mathbf{\text{Hijo}_2^{\text{OX}} = [2, 4, 5, 1, 3, 6, 7, 8]}$$

#### B. Partially Mapped Crossover (PMX) - Paso a Paso
1. **Copiar segmento central**:
   $$\text{Hijo}_1 = [\_, \_, \_, \mathbf{4, 5, 6}, \_, \_]$$
   Mapeo de correspondencias en el segmento de corte:
   $$4 \leftrightarrow 1, \quad 5 \leftrightarrow 3, \quad 6 \leftrightarrow 6$$
2. **Mapear elementos de $P_2[3:6] = [1, 3, 6]$ no presentes en $\{4, 5, 6\}$**:
   - Elemento **1**: En $P_1$ posición 3 está el `4`. El `4` en $P_2$ está en el índice 6 (fuera del corte). $\to \text{Hijo}_1[6] = 1$.
   - Elemento **3**: En $P_1$ posición 4 está el `5`. El `5` en $P_2$ está en el índice 1 (fuera del corte). $\to \text{Hijo}_1[1] = 3$.
   - Elemento **6**: Mapeo identidad $6 \leftrightarrow 6$ (ya presente).
   - Estado intermedio: $[\_, 3, \_, 4, 5, 6, 1, \_]$.
3. **Llenar posiciones vacías directamente de $P_2$**:
   - Posición 0: $P_2[0] = 8$
   - Posición 2: $P_2[2] = 2$
   - Posición 7: $P_2[7] = 7$
4. **Resultados PMX**:
   $$\mathbf{\text{Hijo}_1^{\text{PMX}} = [8, 3, 2, 4, 5, 6, 1, 7]}$$
   $$\mathbf{\text{Hijo}_2^{\text{PMX}} = [4, 2, 5, 1, 3, 6, 7, 8]}$$

#### Comparación:
$$\text{Hijo}_1^{\text{OX}} \ne \text{Hijo}_1^{\text{PMX}} \quad \text{y} \quad \text{Hijo}_2^{\text{OX}} \ne \text{Hijo}_2^{\text{PMX}}$$
Ambos operadores preservan la permutación válida, pero OX prioriza el **orden circular relativo**, mientras que PMX prioriza la **posición absoluta** de los genes.

---

### 3.3 Análisis de Cruces de Aristas: Condición Necesaria vs. Suficiente

#### ¿La mejor ruta encontrada tiene cruces?
> **Respuesta:** **La mejor ruta encontrada (89.73 km) NO TIENE CRUCES (0 cruces).**

```text
Resultados del analizador geométrico (contar_cruces_ruta):
 - Línea Base (Vecino Más Cercano, 108.17 km):  3 cruces.
 - Corrida de Referencia (Semilla 0, 97.65 km): 0 cruces.
 - Mejor Ruta Encontrada (89.73 km):           0 cruces.
```

#### Fundamento Teórico Crítico:
1. **Condición Necesaria pero NO Suficiente**:
   No tener cruces es una condición estrictamente necesaria para el óptimo euclidiano, pero **no es suficiente**. Como evidencia el experimento, la corrida de referencia (semilla 0) tiene **0 cruces** y mide **97.65 km**, siendo casi **8 km peor** que la mejor ruta encontrada (**89.73 km**, también con 0 cruces). Existen muchos ciclos hamiltonianos sin auto-intersecciones en el plano; solo uno es el óptimo global.
2. **Desigualdad Triangular**:
   Por la desigualdad triangular en $\mathbb{R}^2$, la suma de las diagonales de un cuadrilátero convexo siempre es mayor o igual que la suma de cualquiera de sus pares de lados opuestos:
   $$\overline{AC} + \overline{BD} \ge \overline{AB} + \overline{CD}$$
   Deshacer un cruce mediante una reconexión 2-opt **reduce o mantiene la distancia** (la reduce estrictamente siempre que los 4 puntos no sean colineales).
3. **Mecanismo del Operador de Mutación**:
   La mutación por inversión actúa como un operador 2-opt estocástico que desenlaza cruces de aristas eficientemente, permitiendo escapar de configuraciones con cruces hacia ciclos sin cruces.

---

## 4. Batería de Experimentos (Paso 12)

Cada configuración se ejecutó sobre **10 semillas independientes (0 a 9)**. Se midió el mínimo, promedio, desviación estándar y tiempo computacional promedio con `time.perf_counter()`:

| Configuración | Mínimo (km) | Promedio (km) | Desv. Est. (km) | Tiempo Prom (s) |
| :--- | :---: | :---: | :---: | :---: |
| Exp 1: Cruza OX | 89.73 | 95.57 | 3.82 | 0.177 |
| Exp 1: Cruza PMX | 89.73 | 93.33 | 2.72 | 0.186 |
| Exp 2: Mutación 0.01 | 95.78 | 103.91 | 6.85 | 0.135 |
| Exp 2: Mutación 0.05 | 89.73 | 95.57 | 3.82 | 0.169 |
| Exp 2: Mutación 0.20 | 89.73 | 93.99 | 2.55 | 0.168 |
| Exp 3: Población 30 | 89.73 | 99.83 | 5.83 | 0.070 |
| Exp 3: Población 100 | 89.73 | 95.57 | 3.82 | 0.171 |
| Exp 3: Población 300 | 90.70 | 94.32 | 2.45 | 0.482 |
| Exp 4: Sin Elitismo ($n_e=0$) | 91.58 | 95.95 | 2.54 | 0.174 |
| Exp 4: Con Elitismo ($n_e=2$) | 89.73 | 95.57 | 3.82 | 0.170 |
| **Exp 5: Combinada (PMX + Mut 0.20 + Pob 100)** | **89.73** | **92.61** | **1.61** | **0.155** |
| Exp 5: Combinada (PMX + Mut 0.20 + Pob 300) | 89.73 | 92.95 | 2.00 | 0.414 |

### Justificación de la Mejor Configuración (Exp 5):
No se eligió por el mínimo aislado (que depende de la fortuna estocástica de una semilla particular), sino por **promedio y desviación estándar**:
- **Ganadora**: **PMX + Mutación 0.20 + Población 100 + Elitismo 2**.
- **Promedio**: **92.61 km** (el más bajo de todo el estudio).
- **Consistencia**: Desviación estándar de solo **$\pm 1.61\text{ km}$** (la menor variabilidad entre semillas).

### Comparativa con la Línea Base:
$$\text{Mejora} = \frac{L_{\text{NN}} - L_{\text{AG}}}{L_{\text{NN}}} \times 100\%$$
- **Línea Base (Vecino Más Cercano)**: $108.17\text{ km}$
- **Mejora en Promedio del AG**: $\mathbf{+14.39\%}$ ($92.61\text{ km}$)
- **Mejora Máxima Absoluta**: $\mathbf{+17.05\%}$ ($89.73\text{ km}$)

---

## 5. Reto CVRP: Reparto con Restricción de Carga (100 kg)

### 5.1 Mecanismo de Decodificación
Se utilizó un decodificador voraz de partición que recorre la permutación de clientes y abre un nuevo viaje desde $D$ cada vez que la carga acumulada excede los $100\text{ kg}$.

### 5.2 Resultados del CVRP
- **Distancia Total Acumulada**: **105.80 km**
- **Flota Utilizada**: **3 viajes/camionetas** (igual al óptimo teórico $\lceil 270/100 \rceil = 3$).

| Viaje | Tiendas Asignadas | Carga Total (kg) | Porcentaje de Capacidad |
| :---: | :--- | :---: | :---: |
| **Camioneta 1** | P13 (7), P04 (18), P10 (11), P20 (14), P16 (19), P03 (12), P14 (17) | **98.0 kg** | 98.0% |
| **Camioneta 2** | P17 (12), P11 (13), P06 (20), P19 (8), P02 (10), P15 (10), P08 (9) | **82.0 kg** | 82.0% |
| **Camioneta 3** | P12 (22), P05 (8), P18 (15), P09 (16), P01 (15), P07 (14) | **90.0 kg** | 90.0% |

El mapa generado en `ruta_cvrp.png` muestra los tres anillos de entrega independientes saliendo y retornando al depósito central $D(10, 10)$.

---

## 6. Conclusiones

1. Se implementó una solución modular, determinista en sus semillas y 100% nativa en Python, sin dependencias de paquetes de optimización externa.
2. La mutación por inversión (2-opt) demostró ser indispensable en permutaciones espaciales para eliminar cruces de aristas.
3. El operador PMX superó a OX en robustez estadística promedio ($93.33\text{ km}$ vs $95.57\text{ km}$) al preservar información posicional de subrutas.
4. El Algoritmo Genético superó a la heurística constructiva clásica del Vecino Más Cercano en un **17.05%** en su mejor solución y en un **14.39%** en promedio sostenido sobre 10 corridas.
5. El reto CVRP resolvió el empaquetamiento de carga de 270 kg alcanzando el límite físico inferior de 3 viajes con una utilización de capacidad promedio del **90.0%**.

