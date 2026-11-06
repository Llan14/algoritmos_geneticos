"""
=============================================================================
Algoritmo Genético para el Problema del Agente Viajero (TSP) y CVRP
Reparto artesanal de pan en Querétaro
=============================================================================
Implementación modular desde cero (sin bibliotecas externas de optimización
como DEAP o PyGAD) para resolver el TSP de 20 tiendas de abarrotes más el
depósito central (D), con comparación de operadores, análisis estadístico
sobre 10 semillas y extensión al reto Capacitated VRP (CVRP).
"""

import csv
import os
import random
import time
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def obtener_ruta_puntos(ruta_personalizada: Optional[str] = None) -> str:
    """
    Resuelve de forma robusta la ubicación de 'puntos.csv' para garantizar
    que el código corra desde cero en cualquier computadora o directorio de trabajo.

    Parámetros:
        ruta_personalizada (Optional[str]): Ruta explícita si se especifica.

    Retorna:
        str: Ruta existente hacia el archivo CSV de datos.
    """
    if ruta_personalizada and os.path.exists(ruta_personalizada):
        return ruta_personalizada

    # 1. Probar en la misma carpeta donde reside este script
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    ruta_en_script = os.path.join(directorio_script, "puntos.csv")
    if os.path.exists(ruta_en_script):
        return ruta_en_script

    # 2. Probar en el directorio actual de ejecución
    if os.path.exists("puntos.csv"):
        return os.path.abspath("puntos.csv")

    return ruta_personalizada if ruta_personalizada else "puntos.csv"


def resolver_ruta_guardado(nombre_archivo: str) -> str:
    """
    Resuelve la ruta para guardar figuras en el mismo directorio del script,
    asegurando que se generen en la misma carpeta sin importar desde dónde se invoque Python.

    Parámetros:
        nombre_archivo (str): Nombre o ruta relativa del archivo de imagen.

    Retorna:
        str: Ruta absoluta resuelta para guardar la figura.
    """
    if os.path.isabs(nombre_archivo):
        return nombre_archivo
    directorio_script = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(directorio_script, nombre_archivo)


# =============================================================================
# PASO 1: Leer datos y calcular matriz de distancias euclidianas (21 x 21)
# =============================================================================

def leer_puntos(ruta: Optional[str] = None) -> Tuple[List[str], np.ndarray, np.ndarray]:
    """
    Lee las coordenadas y demandas de los nodos desde un archivo CSV.
    Si ruta es None o no se pasa argumento, busca automáticamente 'puntos.csv'
    en la misma carpeta del script para portabilidad garantizada en otra computadora.

    Parámetros:
        ruta (Optional[str]): Ruta opcional al archivo CSV con columnas: id, x_km, y_km, kg.

    Retorna:
        Tuple[List[str], np.ndarray, np.ndarray]:
            - ids: Lista con los identificadores de los nodos ('D', 'P01', ...).
            - coords: Arreglo NumPy (N, 2) con las coordenadas (x, y) en km.
            - dem: Arreglo NumPy (N,) con las demandas de pan en kg.
    """
    ruta_real = obtener_ruta_puntos(ruta)
    ids: List[str] = []
    coords: List[Tuple[float, float]] = []
    dem: List[float] = []
    with open(ruta_real, mode="r", encoding="utf-8") as f:
        lector = csv.DictReader(f)
        for fila in lector:
            ids.append(fila["id"])
            coords.append((float(fila["x_km"]), float(fila["y_km"])))
            dem.append(float(fila["kg"]))
    return ids, np.array(coords, dtype=float), np.array(dem, dtype=float)


def matriz_distancias(coords: np.ndarray) -> np.ndarray:
    """
    Calcula la matriz completa de distancias euclidianas 2D (21 x 21).

    Utiliza el broadcast de NumPy para vectorizar el cálculo:
        d(i, j) = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)

    Parámetros:
        coords (np.ndarray): Matriz de coordenadas de forma (N, 2).

    Retorna:
        np.ndarray: Matriz simétrica de distancias de forma (N, N).
    """
    dif = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((dif ** 2).sum(axis=2))


# =============================================================================
# PASO 2: Representación del cromosoma
# =============================================================================
# El cromosoma es una lista/permutación de los índices 1 a 20 (las 20 tiendas).
# El depósito central 'D' corresponde al índice 0 y no se incluye en el
# cromosoma para evitar redundancia, ya que toda ruta inicia y termina en D.
# Ejemplo de cromosoma: [5, 12, 3, 18, ..., 1] (longitud = 20)


# =============================================================================
# PASO 3: Aptitud y funciones de evaluación
# =============================================================================

def longitud(cromo: List[int], D: np.ndarray) -> float:
    """
    Calcula la longitud total de la ruta en km, cerrando el ciclo en D (índice 0).

    La ruta completa recorrida es:
        0 -> cromo[0] -> cromo[1] -> ... -> cromo[n-1] -> 0

    Parámetros:
        cromo (List[int]): Permutación de los índices de clientes [1..20].
        D (np.ndarray): Matriz de distancias de tamaño (N, N).

    Retorna:
        float: Distancia total del recorrido en kilómetros.
    """
    ruta = [0] + list(cromo) + [0]
    return float(sum(D[ruta[i], ruta[i + 1]] for i in range(len(ruta) - 1)))


def aptitud(cromo: List[int], D: np.ndarray, metodo: str = "inversa", L_max: float = 1000.0) -> float:
    """
    Calcula el valor de aptitud (fitness) a maximizar para un cromosoma.

    Pregunta 7: Implementación de al menos dos formas de aptitud y su diferencia:
      1. 'inversa': f(x) = 1.0 / L(x)
         - Escalamiento hiperbólico no lineal.
                 - En selección por ruleta, la normalización comprime las diferencias
                     entre rutas de longitud similar. Siempre es estrictamente positiva,
                     lo que la hace compatible directamente con selección por ruleta.
      2. 'lineal_complementaria': f(x) = max(0.0, L_max - L(x))
         - Escalamiento lineal donde una mejora de 5 km otorga exactamente el
           mismo incremento de aptitud sin importar la calidad base de la ruta.
           Requiere estimar una cota superior L_max para evitar valores negativos.
      3. 'negativa': f(x) = -L(x)
         - La transformación canónica matemática: minimizar L equivale a
           maximizar -L. Excelente para selección por torneo o ranking donde solo
           importa la relación de orden (f1 > f2), sin necesidad de positividad.

    Parámetros:
        cromo (List[int]): Permutación de clientes.
        D (np.ndarray): Matriz de distancias.
        metodo (str): 'inversa', 'lineal_complementaria' o 'negativa'.
        L_max (float): Cota superior usada en 'lineal_complementaria'.

    Retorna:
        float: Valor numérico de aptitud a maximizar.
    """
    L = longitud(cromo, D)
    if metodo == "inversa":
        return 1.0 / L if L > 0 else float("inf")
    elif metodo == "lineal_complementaria":
        return max(0.0, L_max - L)
    elif metodo == "negativa":
        return -L
    else:
        raise ValueError(f"Método de aptitud '{metodo}' no reconocido.")


# =============================================================================
# PASO 4: Población inicial
# =============================================================================

def poblacion_inicial(tam_pob: int, n: int = 20, sembrar_nn: bool = False,
                      D: Optional[np.ndarray] = None) -> List[List[int]]:
    """
    Genera la población inicial de N permutaciones de los índices 1..n.

    Parámetros:
        tam_pob (int): Tamaño de la población (número de individuos).
        n (int): Número de clientes/tiendas (por defecto 20).
        sembrar_nn (bool): Si es True, incluye 1 individuo generado mediante
            la heurística del vecino más cercano.
        D (Optional[np.ndarray]): Matriz de distancias requerida si sembrar_nn es True.

    Justificación de siembra heurística:
        Sembrar una solución greedy (Vecino Más Cercano) introduce un ancla de
        alta calidad desde la generación 0, acelerando la convergencia inicial.
        Sin embargo, se debe mantener baja la proporción (e.g. 1 o 2 de 100)
        para preservar la diversidad genética y evitar el estancamiento prematuro.

    Retorna:
        List[List[int]]: Lista de individuos (cromosomas).
    """
    poblacion = [random.sample(range(1, n + 1), n) for _ in range(tam_pob)]
    if sembrar_nn and D is not None:
        sol_nn, _ = vecino_mas_cercano(D, n)
        poblacion[0] = sol_nn.copy()
    return poblacion


# =============================================================================
# PASO 5: Operadores de Selección
# =============================================================================

def seleccion(poblacion: List[List[int]], aptitudes: List[float],
              metodo: str = "torneo", k: int = 3) -> List[int]:
    """
    Selecciona un individuo de la población para reproducirse.

    Métodos disponibles:
      - 'torneo': Escoge k individuos al azar y retorna el de mayor aptitud.
                  Robusto, no requiere escalamiento y permite presión ajustable con k.
      - 'ruleta': Selección proporcional a la aptitud (Fitness Proportionate).
                  Requiere aptitudes no negativas.
      - 'ranking': Asigna probabilidades lineales según el orden de aptitud,
                   evitando el dominio de super-individuos o estancamiento.

    Parámetros:
        poblacion (List[List[int]]): Lista de cromosomas actuales.
        aptitudes (List[float]): Lista de valores de aptitud correspondientes.
        metodo (str): 'torneo', 'ruleta' o 'ranking'.
        k (int): Tamaño del torneo si metodo == 'torneo'.

    Retorna:
        List[int]: Copia del cromosoma seleccionado.
    """
    tam = len(poblacion)
    if metodo == "torneo":
        aspirantes = random.sample(range(tam), k)
        mejor_idx = max(aspirantes, key=lambda idx: aptitudes[idx])
        return list(poblacion[mejor_idx])

    elif metodo == "ruleta":
        # Ajustar si hay valores negativos para garantizar probabilidades >= 0
        min_apt = min(aptitudes)
        if min_apt < 0:
            apt_ajustada = [a - min_apt + 1e-6 for a in aptitudes]
        else:
            apt_ajustada = [max(a, 1e-9) for a in aptitudes]
        total = sum(apt_ajustada)
        probs = [a / total for a in apt_ajustada]
        idx = np.random.choice(tam, p=probs)
        return list(poblacion[idx])

    elif metodo == "ranking":
        # Ordenar índices por aptitud ascendente (rango 1 = peor, rango N = mejor)
        indices_ordenados = np.argsort(aptitudes)
        # Probabilidad lineal de ranking: P(r) = 2*r / (N*(N+1))
        pesos = np.arange(1, tam + 1, dtype=float)
        probs = pesos / pesos.sum()
        idx_elegido = np.random.choice(indices_ordenados, p=probs)
        return list(poblacion[idx_elegido])

    else:
        raise ValueError(f"Método de selección '{metodo}' no reconocido.")


# =============================================================================
# PASO 6: Operadores de Cruza (Crossover)
# =============================================================================

def cruza_ox(p1: List[int], p2: List[int],
             puntos_corte: Optional[Tuple[int, int]] = None) -> Tuple[List[int], List[int]]:
    """
    Operador Order Crossover (OX) para permutaciones.

    Esquema:
      1. Elige dos puntos de corte a < b (o usa puntos_corte si se especifica).
      2. El hijo 1 copia el segmento p1[a:b] en las mismas posiciones.
      3. Los huecos restantes se llenan con los genes de p2, recorriéndolo
         en orden circular a partir de b, omitiendo los genes ya presentes.
      4. Se repite simétricamente para el hijo 2 intercambiando roles de p1 y p2.

    Preserva el orden relativo y la contigüidad de sub-rutas.
    """
    n = len(p1)
    if puntos_corte is not None:
        a, b = puntos_corte
    else:
        a, b = sorted(random.sample(range(n + 1), 2))
        while a == b or (a == 0 and b == n):
            a, b = sorted(random.sample(range(n + 1), 2))

    def _ox_un_hijo(padre_a: List[int], padre_b: List[int]) -> List[int]:
        """Genera un hijo OX: copia padre_a[a:b] y rellena huecos con padre_b en orden circular desde b."""
        hijo = [None] * n
        # Copiar segmento
        hijo[a:b] = padre_a[a:b]
        presentes = set(padre_a[a:b])

        # Recorrer padre_b circularmente desde b
        genes_b = [padre_b[(b + i) % n] for i in range(n)]
        restantes = [g for g in genes_b if g not in presentes]

        # Llenar huecos circularmente desde b
        idx_restante = 0
        for i in range(n):
            pos = (b + i) % n
            if hijo[pos] is None:
                hijo[pos] = restantes[idx_restante]
                idx_restante += 1
        return hijo

    h1 = _ox_un_hijo(p1, p2)
    h2 = _ox_un_hijo(p2, p1)
    return h1, h2


def cruza_pmx(p1: List[int], p2: List[int],
              puntos_corte: Optional[Tuple[int, int]] = None) -> Tuple[List[int], List[int]]:
    """
    Operador Partially Mapped Crossover (PMX) para permutaciones.

    Esquema:
      1. Selecciona dos puntos de corte a < b (o usa puntos_corte si se especifica).
      2. Copia el segmento p1[a:b] al hijo.
      3. Para cada elemento en p2[a:b] que no esté en el segmento copiado,
         busca su posición en p2 fuera del segmento siguiendo la relación de
         mapeo uno a uno establecida entre p1[a:b] y p2[a:b].
      4. Llena las posiciones restantes directamente de p2.
      5. Se repite simétricamente para generar el hijo 2.

    Preserva la posición absoluta de la mayor cantidad posible de genes.
    """
    n = len(p1)
    if puntos_corte is not None:
        a, b = puntos_corte
    else:
        a, b = sorted(random.sample(range(n + 1), 2))
        while a == b or (a == 0 and b == n):
            a, b = sorted(random.sample(range(n + 1), 2))

    def _pmx_un_hijo(padre_a: List[int], padre_b: List[int]) -> List[int]:
        """Genera un hijo PMX: copia padre_a[a:b], resuelve conflictos por mapeo y completa con padre_b."""
        hijo = [None] * n
        # 1. Copiar segmento de mapeo
        hijo[a:b] = padre_a[a:b]

        # 2. Mapear elementos de padre_b[a:b]
        for i in range(a, b):
            val_b = padre_b[i]
            if val_b not in padre_a[a:b]:
                pos = i
                while a <= pos < b:
                    val_a = padre_a[pos]
                    pos = padre_b.index(val_a)
                hijo[pos] = val_b

        # 3. Llenar los vacíos restantes directamente desde padre_b
        for i in range(n):
            if hijo[i] is None:
                hijo[i] = padre_b[i]
        return hijo

    h1 = _pmx_un_hijo(p1, p2)
    h2 = _pmx_un_hijo(p2, p1)
    return h1, h2


def cruza(p1: List[int], p2: List[int], metodo: str = "OX",
          puntos_corte: Optional[Tuple[int, int]] = None) -> Tuple[List[int], List[int]]:
    """
    Despachador para operadores de cruza.

    Parámetros:
        p1 (List[int]): Padre 1.
        p2 (List[int]): Padre 2.
        metodo (str): 'OX' (Order Crossover) o 'PMX' (Partially Mapped Crossover).
        puntos_corte (Optional[Tuple[int, int]]): Puntos de corte opcionales (a, b).

    Retorna:
        Tuple[List[int], List[int]]: Dos hijos válidos generados.
    """
    if metodo.upper() == "OX":
        return cruza_ox(p1, p2, puntos_corte=puntos_corte)
    elif metodo.upper() == "PMX":
        return cruza_pmx(p1, p2, puntos_corte=puntos_corte)
    else:
        raise ValueError(f"Método de cruza '{metodo}' no reconocido (use 'OX' o 'PMX').")


# =============================================================================
# PASO 7: Operadores de Mutación
# =============================================================================

def mutacion_swap(cromo: List[int]) -> List[int]:
    """
    Mutación por intercambio (Swap Mutation):
    Selecciona dos posiciones al azar e intercambia sus valores.
    Perturbación mínima que explora vecinos inmediatos.
    """
    c = list(cromo)
    i, j = random.sample(range(len(c)), 2)
    c[i], c[j] = c[j], c[i]
    return c


def mutacion_inversion(cromo: List[int]) -> List[int]:
    """
    Mutación por inversión (Inversion / 2-opt Mutation):
    Selecciona un segmento cromo[a:b+1] y lo invierte:
        cromo[a:b+1] = cromo[a:b+1][::-1]
    Equivale al movimiento clásico 2-opt en TSP, eliminando cruces de aristas.
    """
    c = list(cromo)
    i, j = sorted(random.sample(range(len(c)), 2))
    c[i:j + 1] = c[i:j + 1][::-1]
    return c


def mutacion(cromo: List[int], tasa: float, tipo: str = "inversion") -> List[int]:
    """
    Aplica mutación al cromosoma con probabilidad 'tasa'.

    Parámetros:
        cromo (List[int]): Cromosoma original.
        tasa (float): Probabilidad de aplicar mutación (0.0 a 1.0).
        tipo (str): 'inversion' (2-opt) o 'swap'.

    Retorna:
        List[int]: Cromosoma (mutado o idéntico).
    """
    if random.random() < tasa:
        if tipo == "inversion":
            return mutacion_inversion(cromo)
        elif tipo == "swap":
            return mutacion_swap(cromo)
        else:
            raise ValueError(f"Tipo de mutación '{tipo}' no reconocido.")
    return list(cromo)


# =============================================================================
# PASO 8: Validación de Cromosomas
# =============================================================================

def es_valido(cromo: List[int], n: int = 20) -> bool:
    """
    Valida que el cromosoma sea una permutación estricta de 1 a n
    (sin repeticiones, sin omitir ningún cliente y con longitud n).

    Parámetros:
        cromo (List[int]): Cromosoma a verificar.
        n (int): Cantidad de clientes (20).

    Retorna:
        bool: True si es una solución factible, False en caso contrario.
    """
    return sorted(cromo) == list(range(1, n + 1))


# =============================================================================
# PASO 9: Ciclo Principal del Algoritmo Genético (AG)
# =============================================================================

def ag(D: np.ndarray,
       tam_pob: int = 100,
       p_cruza: float = 0.9,
       p_mut: float = 0.05,
       n_elite: int = 2,
       max_gen: int = 300,
       paciencia: int = 60,
       semilla: int = 0,
       metodo_cruza: str = "OX",
       metodo_mutacion: str = "inversion",
       metodo_seleccion: str = "torneo",
       k_torneo: int = 3,
       metodo_aptitud: str = "inversa",
       sembrar_nn: bool = False,
       n_puntos: int = 20) -> Tuple[List[int], float, List[float], List[float]]:
    """
    Ejecuta el Algoritmo Genético canónico para el TSP con elitismo y parada temprana.

    Parámetros:
        D (np.ndarray): Matriz de distancias (N, N).
        tam_pob (int): Tamaño de la población (ej. 30, 100, 300).
        p_cruza (float): Probabilidad de cruza.
        p_mut (float): Probabilidad de mutación por individuo.
        n_elite (int): Número de mejores individuos copiados intactos (elitismo).
        max_gen (int): Límite máximo de generaciones.
        paciencia (int): Generaciones consecutivas sin mejora antes del paro temprano.
        semilla (int): Semilla para random y numpy (reproducibilidad).
        metodo_cruza (str): 'OX' o 'PMX'.
        metodo_mutacion (str): 'inversion' o 'swap'.
        metodo_seleccion (str): 'torneo', 'ruleta' o 'ranking'.
        k_torneo (int): Participantes en torneo.
        metodo_aptitud (str): 'inversa', 'lineal_complementaria' o 'negativa'.
        sembrar_nn (bool): Siembra la solución del vecino más cercano en la pob. inicial.
        n_puntos (int): Cantidad de clientes (20).

    Retorna:
        Tuple[List[int], float, List[float], List[float]]:
            - mejor_cromo: Mejor cromosoma encontrado.
            - mejor_dist: Distancia mínima lograda en km.
            - historial_mejor: Evolución de la mejor distancia por generación.
            - historial_prom: Evolución de la distancia promedio por generación.
    """
    random.seed(semilla)
    np.random.seed(semilla)

    # 1. Población inicial y validación
    pob = poblacion_inicial(tam_pob, n=n_puntos, sembrar_nn=sembrar_nn, D=D)
    for ind in pob:
        assert es_valido(ind, n_puntos), "Individuo inicial inválido."

    historial_mejor: List[float] = []
    historial_prom: List[float] = []

    mejor_cromo_global: List[int] = list(pob[0])
    mejor_dist_global: float = float("inf")
    gen_sin_mejora = 0

    # 2. Ciclo generacional
    for gen in range(max_gen):
        # Evaluar distancias y aptitudes
        distancias = [longitud(ind, D) for ind in pob]
        aptitudes = [aptitud(ind, D, metodo=metodo_aptitud) for ind in pob]

        # Estadísticas de la generación
        gen_mejor_dist = min(distancias)
        gen_prom_dist = float(np.mean(distancias))
        historial_mejor.append(gen_mejor_dist)
        historial_prom.append(gen_prom_dist)

        # Actualizar mejor global
        if gen_mejor_dist < mejor_dist_global - 1e-5:
            mejor_dist_global = gen_mejor_dist
            idx_mejor = int(np.argmin(distancias))
            mejor_cromo_global = list(pob[idx_mejor])
            gen_sin_mejora = 0
        else:
            gen_sin_mejora += 1

        # Criterio de parada temprana por paciencia
        if gen_sin_mejora >= paciencia:
            break

        # 3. Construcción de la nueva generación
        nueva_pob: List[List[int]] = []

        # Elitismo: copiar los n_elite mejores directamente
        if n_elite > 0:
            indices_ordenados = np.argsort(distancias)  # Menor distancia es mejor
            for i in range(min(n_elite, tam_pob)):
                nueva_pob.append(list(pob[indices_ordenados[i]]))

        # Reproducción (Selección -> Cruza -> Mutación -> Validación)
        while len(nueva_pob) < tam_pob:
            p1 = seleccion(pob, aptitudes, metodo=metodo_seleccion, k=k_torneo)
            p2 = seleccion(pob, aptitudes, metodo=metodo_seleccion, k=k_torneo)

            if random.random() < p_cruza:
                h1, h2 = cruza(p1, p2, metodo=metodo_cruza)
            else:
                h1, h2 = list(p1), list(p2)

            h1 = mutacion(h1, tasa=p_mut, tipo=metodo_mutacion)
            h2 = mutacion(h2, tasa=p_mut, tipo=metodo_mutacion)

            assert es_valido(h1, n_puntos), f"Hijo 1 inválido tras cruza {metodo_cruza}/mutación"
            assert es_valido(h2, n_puntos), f"Hijo 2 inválido tras cruza {metodo_cruza}/mutación"

            nueva_pob.append(h1)
            if len(nueva_pob) < tam_pob:
                nueva_pob.append(h2)

        pob = nueva_pob

    return mejor_cromo_global, mejor_dist_global, historial_mejor, historial_prom


# =============================================================================
# PASO 10: Heurística del Vecino Más Cercano (Línea Base determinista)
# =============================================================================

def vecino_mas_cercano(D: np.ndarray, n: int = 20) -> Tuple[List[int], float]:
    """
    Construye una ruta greedy determinista partiendo de D (índice 0),
    visitando siempre el nodo no visitado más cercano y regresando a D.

    Parámetros:
        D (np.ndarray): Matriz de distancias (21 x 21).
        n (int): Cantidad de tiendas (20).

    Retorna:
        Tuple[List[int], float]:
            - cromo_nn: Permutación de tiendas visitadas [1..n].
            - dist_nn: Distancia total del recorrido en km.
    """
    no_visitados = set(range(1, n + 1))
    actual = 0
    cromo_nn: List[int] = []

    while no_visitados:
        siguiente = min(no_visitados, key=lambda nodo: D[actual, nodo])
        cromo_nn.append(siguiente)
        no_visitados.remove(siguiente)
        actual = siguiente

    dist_nn = longitud(cromo_nn, D)
    return cromo_nn, dist_nn


def calcular_mejora(dist_nn: float, dist_ag: float) -> float:
    """
    Calcula el porcentaje de mejora del AG respecto a la línea base:
        mejora = (L_NN - L_AG) / L_NN * 100
    """
    return ((dist_nn - dist_ag) / dist_nn) * 100.0


# =============================================================================
# PASO 11: Gráficas de Resultados (Convergencia y Ruta Geográfica)
# =============================================================================

def graficar_convergencia(hist_mejor: List[float],
                          hist_prom: List[float],
                          titulo: str = "Convergencia del Algoritmo Genético",
                          guardar: Optional[str] = "convergencia_tsp.png") -> None:
    """
    Genera la gráfica de convergencia mostrando la mejor distancia y la
    distancia promedio de la población a través de las generaciones.
    """
    plt.figure(figsize=(9, 5))
    generaciones = range(len(hist_mejor))
    plt.plot(generaciones, hist_mejor, label="Mejor individuo ($L_{min}$)", color="#1b5e20", linewidth=2.2)
    plt.plot(generaciones, hist_prom, label="Promedio de población ($L_{prom}$)", color="#0277bd", linestyle="--", linewidth=1.6)

    plt.title(titulo, fontsize=13, fontweight="bold")
    plt.xlabel("Generación", fontsize=11)
    plt.ylabel("Distancia total de la ruta (km)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=10)
    plt.tight_layout()

    if guardar:
        plt.savefig(resolver_ruta_guardado(guardar), dpi=300)
    plt.close()


def contar_cruces_ruta(cromo: List[int], coords: np.ndarray) -> int:
    """
    Cuenta el número de cruces/intersecciones de aristas en una ruta TSP 2D.
    En una ruta euclidiana óptima, el número de cruces debe ser 0 (desigualdad triangular).

    Parámetros:
        cromo (List[int]): Permutación de clientes.
        coords (np.ndarray): Coordenadas (N, 2).

    Retorna:
        int: Cantidad de intersecciones detectadas entre segmentos no adyacentes.
    """
    def _orientacion(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> int:
        """Determina la orientación del triplete de puntos (colineal, horario o antihorario)."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-9:
            return 0
        return 1 if val > 0 else 2

    def _en_segmento(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        """Comprueba si el punto q yace sobre el segmento pr sabiendo que son colineales."""
        return (q[0] <= max(p[0], r[0]) + 1e-9 and q[0] >= min(p[0], r[0]) - 1e-9 and
                q[1] <= max(p[1], r[1]) + 1e-9 and q[1] >= min(p[1], r[1]) - 1e-9)

    def _se_cruzan(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> bool:
        """Determina si dos segmentos p1q1 y p2q2 se intersectan en el plano cartesiano."""
        o1 = _orientacion(p1, q1, p2)
        o2 = _orientacion(p1, q1, q2)
        o3 = _orientacion(p2, q2, p1)
        o4 = _orientacion(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and _en_segmento(p1, p2, q1): return True
        if o2 == 0 and _en_segmento(p1, q2, q1): return True
        if o3 == 0 and _en_segmento(p2, p1, q2): return True
        if o4 == 0 and _en_segmento(p2, q1, q2): return True
        return False

    ruta = [0] + list(cromo) + [0]
    n_aristas = len(ruta) - 1
    cruces = 0
    for i in range(n_aristas):
        p1 = coords[ruta[i]]
        q1 = coords[ruta[i + 1]]
        for j in range(i + 2, n_aristas):
            if i == 0 and j == n_aristas - 1:
                continue
            p2 = coords[ruta[j]]
            q2 = coords[ruta[j + 1]]
            if _se_cruzan(p1, q1, p2, q2):
                cruces += 1
    return cruces


def graficar_ruta(cromo: List[int],
                  coords: np.ndarray,
                  ids: List[str],
                  titulo: str = "Mejor Ruta de Reparto (TSP)",
                  guardar: Optional[str] = "ruta_tsp.png") -> None:
    """
    Dibuja el mapa de Querétaro con la ruta D -> P... -> D,
    destacando el depósito D y enumerando la secuencia de visitas.
    """
    plt.figure(figsize=(9, 8))
    ruta_indices = [0] + list(cromo) + [0]
    puntos_x = coords[ruta_indices, 0]
    puntos_y = coords[ruta_indices, 1]

    # Dibujar líneas del recorrido con flechas sutiles
    plt.plot(puntos_x, puntos_y, color="#1565c0", linewidth=1.8, linestyle="-", zorder=2, alpha=0.85)

    # Dibujar flechas para indicar dirección
    for i in range(len(ruta_indices) - 1):
        x1, y1 = puntos_x[i], puntos_y[i]
        x2, y2 = puntos_x[i + 1], puntos_y[i + 1]
        dx, dy = (x2 - x1) * 0.25, (y2 - y1) * 0.25
        plt.annotate("", xy=(x1 + dx + (x2 - x1) * 0.5, y1 + dy + (y2 - y1) * 0.5),
                     xytext=(x1 + (x2 - x1) * 0.5, y1 + (y2 - y1) * 0.5),
                     arrowprops=dict(arrowstyle="->", color="#0d47a1", lw=1.2), zorder=3)

    # Tiendas
    tiendas_idx = range(1, len(ids))
    plt.scatter(coords[tiendas_idx, 0], coords[tiendas_idx, 1],
                color="#e65100", s=110, edgecolor="black", linewidth=1.2, zorder=4, label="Tiendas (P01..P20)")

    # Depósito Central (D)
    plt.scatter(coords[0, 0], coords[0, 1],
                color="#d50000", s=250, marker="s", edgecolor="black", linewidth=2.0, zorder=5, label="Depósito D (Inicio/Fin)")

    # Etiquetas con IDs y orden de visita
    orden_visita = {cromo[i]: i + 1 for i in range(len(cromo))}
    plt.annotate(f"D (10, 10)\nInicio/Fin", (coords[0, 0] + 0.35, coords[0, 1] + 0.25),
                 fontsize=9.5, fontweight="bold", color="#b71c1c", zorder=6)

    for i in tiendas_idx:
        sec = orden_visita.get(i, "")
        lbl = f"{ids[i]} (#{sec})"
        plt.annotate(lbl, (coords[i, 0] + 0.25, coords[i, 1] + 0.2),
                     fontsize=8.5, color="#263238", zorder=6)

    plt.title(titulo, fontsize=13, fontweight="bold")
    plt.xlabel("Coordenada X (km)", fontsize=11)
    plt.ylabel("Coordenada Y (km)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    plt.tight_layout()

    if guardar:
        plt.savefig(resolver_ruta_guardado(guardar), dpi=300)
    plt.close()


# =============================================================================
# PASO 12: Batería de Experimentos (10 semillas por configuración)
# =============================================================================

def ejecutar_experimento(D: np.ndarray,
                         configuraciones: Dict[str, Dict[str, Any]],
                         semillas: range = range(10)) -> Tuple[pd.DataFrame, Dict[str, List[float]], Dict[str, Any]]:
    """
    Ejecuta múltiples configuraciones del AG sobre 10 semillas (0 a 9).

    Calcula para cada configuración:
      - Mejor (mínimo global de distancia)
      - Promedio de distancias
      - Desviación estándar (np.std)
      - Tiempo promedio de ejecución por semilla en segundos (time.perf_counter)
      - Guarda el mejor cromosoma absoluto y su historial de convergencia.

    Retorna:
        Tuple[pd.DataFrame, Dict[str, List[float]], Dict[str, Any]]:
            - df_resumen: Tabla resumen de pandas.
            - datos_boxplot: Diccionario {nombre_config: [distancias_10_semillas]}.
            - info_mejores: Diccionario con el mejor cromosoma global, su distancia,
                            historiales y mejores por configuración.
    """
    resumen_filas = []
    datos_boxplot: Dict[str, List[float]] = {}
    mejores_por_config: Dict[str, Dict[str, Any]] = {}

    mejor_global_dist = float("inf")
    mejor_global_cromo: Optional[List[int]] = None
    mejor_global_hist_min: List[float] = []
    mejor_global_hist_prom: List[float] = []
    mejor_global_config = ""
    mejor_global_semilla = -1

    for nombre, params in configuraciones.items():
        distancias_semillas = []
        tiempos = []

        cfg_mejor_dist = float("inf")
        cfg_mejor_cromo: Optional[List[int]] = None
        cfg_mejor_hist_min: List[float] = []
        cfg_mejor_hist_prom: List[float] = []
        cfg_mejor_semilla = -1

        for sem in semillas:
            t0 = time.perf_counter()
            cromo, dist, h_min, h_prom = ag(D, semilla=sem, **params)
            t1 = time.perf_counter()

            distancias_semillas.append(dist)
            tiempos.append(t1 - t0)

            if dist < cfg_mejor_dist:
                cfg_mejor_dist = dist
                cfg_mejor_cromo = list(cromo)
                cfg_mejor_hist_min = list(h_min)
                cfg_mejor_hist_prom = list(h_prom)
                cfg_mejor_semilla = sem

            if dist < mejor_global_dist:
                mejor_global_dist = dist
                mejor_global_cromo = list(cromo)
                mejor_global_hist_min = list(h_min)
                mejor_global_hist_prom = list(h_prom)
                mejor_global_config = nombre
                mejor_global_semilla = sem

        mejores_por_config[nombre] = {
            "cromo": cfg_mejor_cromo,
            "distancia": cfg_mejor_dist,
            "hist_min": cfg_mejor_hist_min,
            "hist_prom": cfg_mejor_hist_prom,
            "semilla": cfg_mejor_semilla
        }

        datos_boxplot[nombre] = distancias_semillas
        resumen_filas.append({
            "Configuración": nombre,
            "Mínimo (km)": round(float(np.min(distancias_semillas)), 2),
            "Promedio (km)": round(float(np.mean(distancias_semillas)), 2),
            "Desv. Est. (km)": round(float(np.std(distancias_semillas)), 2),
            "Tiempo Prom (s)": round(float(np.mean(tiempos)), 3)
        })

    df_resumen = pd.DataFrame(resumen_filas)
    info_mejores = {
        "mejor_global_cromo": mejor_global_cromo,
        "mejor_global_dist": mejor_global_dist,
        "mejor_global_hist_min": mejor_global_hist_min,
        "mejor_global_hist_prom": mejor_global_hist_prom,
        "mejor_global_config": mejor_global_config,
        "mejor_global_semilla": mejor_global_semilla,
        "mejores_por_config": mejores_por_config
    }
    return df_resumen, datos_boxplot, info_mejores


def graficar_boxplots(datos_boxplot: Dict[str, List[float]],
                      titulo: str = "Comparación de Configuraciones del AG (10 Semillas)",
                      guardar: Optional[str] = "boxplots_experimentos.png") -> None:
    """
    Genera un diagrama de caja (boxplot) comparando la distribución de
    distancias logradas por cada configuración probada.
    """
    plt.figure(figsize=(11, 6))
    etiquetas = list(datos_boxplot.keys())
    valores = [datos_boxplot[k] for k in etiquetas]

    bp = plt.boxplot(valores, tick_labels=etiquetas, patch_artist=True,
                     boxprops=dict(facecolor="#bbdefb", color="#0d47a1"),
                     whiskerprops=dict(color="#0d47a1", linewidth=1.5),
                     capprops=dict(color="#0d47a1", linewidth=1.5),
                     medianprops=dict(color="#d50000", linewidth=2.0),
                     flierprops=dict(marker="o", markerfacecolor="#e65100", markersize=6))

    plt.title(titulo, fontsize=13, fontweight="bold")
    plt.ylabel("Distancia de la mejor ruta (km)", fontsize=11)
    plt.xticks(rotation=25, ha="right", fontsize=9.5)
    plt.grid(True, axis="y", linestyle=":", alpha=0.6)
    plt.tight_layout()

    if guardar:
        plt.savefig(resolver_ruta_guardado(guardar), dpi=300)
    plt.close()


# =============================================================================
# RETO EXTRA: CVRP (Capacitated Vehicle Routing Problem)
# =============================================================================

def decodificar_cvrp(cromo: List[int],
                     demandas: np.ndarray,
                     capacidad_max: float = 100.0) -> List[List[int]]:
    """
    Decodifica una permutación de clientes en múltiples viajes de camioneta.

    Algoritmo voraz de partición secuencial:
      - Recorre el cromosoma en orden.
      - Acumula la demanda en el viaje actual.
      - Si agregar la siguiente tienda excede la capacidad de la camioneta (100 kg),
        cierra el viaje actual (regresando a D) e inicia un nuevo viaje saliendo de D.
      - Retorna la lista de viajes, donde cada viaje es una sub-ruta [p_i, ..., p_j].

    Parámetros:
        cromo (List[int]): Permutación de clientes [1..20].
        demandas (np.ndarray): Demandas de pan en kg para cada nodo (índice 0 = 0 kg).
        capacidad_max (float): Capacidad máxima de la camioneta (100 kg).

    Retorna:
        List[List[int]]: Lista de viajes individuales.
    """
    viajes: List[List[int]] = []
    viaje_actual: List[int] = []
    carga_actual = 0.0

    for tienda in cromo:
        d = demandas[tienda]
        if carga_actual + d <= capacidad_max:
            viaje_actual.append(tienda)
            carga_actual += d
        else:
            if viaje_actual:
                viajes.append(viaje_actual)
            viaje_actual = [tienda]
            carga_actual = d

    if viaje_actual:
        viajes.append(viaje_actual)

    return viajes


def longitud_cvrp(cromo: List[int],
                  D: np.ndarray,
                  demandas: np.ndarray,
                  capacidad_max: float = 100.0) -> float:
    """
    Calcula la distancia total acumulada de todos los viajes del CVRP.

    Cada viaje k inicia en D (0), recorre su sub-ruta y regresa a D (0):
        L_k = D[0, t_0] + sum(D[t_i, t_{i+1}]) + D[t_{-1}, 0]
        L_total = sum(L_k)
    """
    viajes = decodificar_cvrp(cromo, demandas, capacidad_max)
    distancia_total = 0.0

    for viaje in viajes:
        ruta = [0] + viaje + [0]
        for i in range(len(ruta) - 1):
            distancia_total += float(D[ruta[i], ruta[i + 1]])

    return distancia_total


def ag_cvrp(D: np.ndarray,
            demandas: np.ndarray,
            capacidad_max: float = 100.0,
            tam_pob: int = 150,
            p_cruza: float = 0.9,
            p_mut: float = 0.08,
            n_elite: int = 2,
            max_gen: int = 300,
            paciencia: int = 70,
            semilla: int = 42) -> Tuple[List[int], float, List[List[int]]]:
    """
    Resuelve el reto CVRP reutilizando la estructura genética del TSP,
    sustituyendo la función de costo por 'longitud_cvrp'.
    """
    random.seed(semilla)
    np.random.seed(semilla)
    n_puntos = len(demandas) - 1

    pob = [random.sample(range(1, n_puntos + 1), n_puntos) for _ in range(tam_pob)]
    mejor_cromo = list(pob[0])
    mejor_dist = float("inf")
    gen_sin_mejora = 0

    for gen in range(max_gen):
        distancias = [longitud_cvrp(ind, D, demandas, capacidad_max) for ind in pob]
        aptitudes = [1.0 / d for d in distancias]

        min_d = min(distancias)
        if min_d < mejor_dist - 1e-5:
            mejor_dist = min_d
            mejor_cromo = list(pob[int(np.argmin(distancias))])
            gen_sin_mejora = 0
        else:
            gen_sin_mejora += 1

        if gen_sin_mejora >= paciencia:
            break

        nueva_pob = []
        if n_elite > 0:
            indices_ordenados = np.argsort(distancias)
            for i in range(n_elite):
                nueva_pob.append(list(pob[indices_ordenados[i]]))

        while len(nueva_pob) < tam_pob:
            p1 = seleccion(pob, aptitudes, metodo="torneo", k=3)
            p2 = seleccion(pob, aptitudes, metodo="torneo", k=3)

            if random.random() < p_cruza:
                h1, h2 = cruza_ox(p1, p2)
            else:
                h1, h2 = list(p1), list(p2)

            h1 = mutacion_inversion(h1) if random.random() < p_mut else h1
            h2 = mutacion_inversion(h2) if random.random() < p_mut else h2

            nueva_pob.append(h1)
            if len(nueva_pob) < tam_pob:
                nueva_pob.append(h2)

        pob = nueva_pob

    mejores_viajes = decodificar_cvrp(mejor_cromo, demandas, capacidad_max)
    return mejor_cromo, mejor_dist, mejores_viajes


def graficar_cvrp(viajes: List[List[int]],
                  coords: np.ndarray,
                  ids: List[str],
                  demandas: np.ndarray,
                  titulo: str = "Rutas Multiviaje CVRP (Capacidad 100 kg)",
                  guardar: Optional[str] = "ruta_cvrp.png") -> None:
    """
    Dibuja los múltiples viajes de camioneta en colores distintos,
    indicando la carga total de cada viaje y las tiendas asignadas.
    """
    plt.figure(figsize=(10, 8.5))
    colores_viajes = ["#1976d2", "#388e3c", "#d32f2f", "#7b1fa2", "#f57c00"]

    # Dibujar cada viaje
    for k, viaje in enumerate(viajes):
        color = colores_viajes[k % len(colores_viajes)]
        ruta_idx = [0] + viaje + [0]
        px = coords[ruta_idx, 0]
        py = coords[ruta_idx, 1]
        carga_viaje = sum(demandas[t] for t in viaje)

        lbl = f"Camioneta {k+1}: {len(viaje)} tiendas, {carga_viaje:.0f} kg"
        plt.plot(px, py, color=color, linewidth=2.0, linestyle="-", alpha=0.85, label=lbl, zorder=2)

    # Tiendas
    tiendas_idx = range(1, len(ids))
    plt.scatter(coords[tiendas_idx, 0], coords[tiendas_idx, 1],
                color="#ffa000", s=110, edgecolor="black", linewidth=1.2, zorder=4)

    # Depósito
    plt.scatter(coords[0, 0], coords[0, 1],
                color="#b71c1c", s=250, marker="s", edgecolor="black", linewidth=2.0, zorder=5, label="Depósito D")

    # Anotaciones
    plt.annotate("D (Depósito)", (coords[0, 0] + 0.35, coords[0, 1] + 0.25),
                 fontsize=10, fontweight="bold", color="#b71c1c", zorder=6)

    for i in tiendas_idx:
        plt.annotate(f"{ids[i]} ({demandas[i]:.0f}kg)",
                     (coords[i, 0] + 0.25, coords[i, 1] + 0.2),
                     fontsize=8.0, color="#263238", zorder=6)

    plt.title(titulo, fontsize=13, fontweight="bold")
    plt.xlabel("Coordenada X (km)", fontsize=11)
    plt.ylabel("Coordenada Y (km)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#cccccc", fontsize=9.5)
    plt.tight_layout()

    if guardar:
        plt.savefig(resolver_ruta_guardado(guardar), dpi=300)
    plt.close()


# =============================================================================
# EJEMPLO DIDÁCTICO DE 8 GENES (Pregunta 10)
# =============================================================================

def probar_ejemplo_8_genes(puntos_corte: Tuple[int, int] = (3, 6)) -> Dict[str, Any]:
    """
    Ejecuta y verifica paso a paso un ejemplo canónico de 8 genes para OX y PMX,
    con un corte multigen no trivial (por defecto posiciones 3 a 5, corte [3:6]),
    demostrando que ambos operadores producen hijos válidos y marcadamente distintos.
    """
    p1 = [1, 2, 3, 4, 5, 6, 7, 8]
    p2 = [8, 5, 2, 1, 3, 6, 4, 7]

    h1_ox, h2_ox = cruza_ox(p1, p2, puntos_corte=puntos_corte)
    h1_pmx, h2_pmx = cruza_pmx(p1, p2, puntos_corte=puntos_corte)

    val_ox1 = es_valido(h1_ox, 8)
    val_ox2 = es_valido(h2_ox, 8)
    val_pmx1 = es_valido(h1_pmx, 8)
    val_pmx2 = es_valido(h2_pmx, 8)

    return {
        "padre1": p1,
        "padre2": p2,
        "puntos_corte": puntos_corte,
        "hijo1_ox": h1_ox,
        "hijo2_ox": h2_ox,
        "valido_ox": val_ox1 and val_ox2,
        "hijo1_pmx": h1_pmx,
        "hijo2_pmx": h2_pmx,
        "valido_pmx": val_pmx1 and val_pmx2
    }


if __name__ == "__main__":
    print("Módulo 'ag_tsp.py' cargado correctamente.")

