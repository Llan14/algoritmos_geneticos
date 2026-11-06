"""
Pruebas unitarias y verificación paso a paso del ejemplo de 8 genes
para la Pregunta 10 (Operadores OX y PMX calculados a mano y por código).
"""

from typing import List, Tuple
from ag_tsp import es_valido


def cruza_ox_paso_a_paso(padre_a: List[int], padre_b: List[int], a: int, b: int,
                         nombre_a: str = "p1", nombre_b: str = "p2") -> Tuple[List[int], List[str]]:
    """
    Ejecuta el operador OX con puntos de corte fijos a y b, registrando
    cada paso de la traza para contrastar con el cálculo hecho a mano.
    Permite etiquetar dinámicamente qué padre aporta el segmento copiado
    y cuál aporta los genes circulares.
    """
    n = len(padre_a)
    hijo = [None] * n
    traza = []

    # 1. Copiar segmento del primer padre
    hijo[a:b] = padre_a[a:b]
    presentes = set(padre_a[a:b])
    traza.append(f"Paso 1: Copiar {nombre_a}[{a}:{b}] = {padre_a[a:b]} en las mismas posiciones.")
    traza.append(f"        Estado del hijo: {hijo}")

    # 2. Recorrer el segundo padre circularmente a partir de b
    genes_b_circ = [padre_b[(b + i) % n] for i in range(n)]
    traza.append(f"Paso 2: Genes de {nombre_b} en orden circular desde índice {b}: {genes_b_circ}")

    # 3. Filtrar genes ya presentes en el segmento copiado
    restantes = [g for g in genes_b_circ if g not in presentes]
    traza.append(f"Paso 3: Genes restantes tras filtrar {sorted(presentes)}: {restantes}")

    # 4. Llenar los huecos circularmente desde b
    idx_r = 0
    for i in range(n):
        pos = (b + i) % n
        if hijo[pos] is None:
            hijo[pos] = restantes[idx_r]
            traza.append(f"        Asignar hijo[{pos}] = {restantes[idx_r]}")
            idx_r += 1

    traza.append(f"Paso 4: Hijo final resultante: {hijo}")
    return hijo, traza


def cruza_pmx_paso_a_paso(padre_a: List[int], padre_b: List[int], a: int, b: int,
                          nombre_a: str = "p1", nombre_b: str = "p2") -> Tuple[List[int], List[str]]:
    """
    Ejecuta el operador PMX con puntos de corte fijos a y b, registrando
    la tabla de correspondencias y la resolución de mapeos.
    """
    n = len(padre_a)
    hijo = [None] * n
    traza = []

    # 1. Copiar segmento del primer padre
    hijo[a:b] = padre_a[a:b]
    traza.append(f"Paso 1: Copiar {nombre_a}[{a}:{b}] = {padre_a[a:b]}.")
    traza.append(f"        Estado del hijo: {hijo}")

    # Mapeo posicional entre ambos padres dentro del segmento de corte
    mapeo_str = [f"{padre_a[i]} <-> {padre_b[i]}" for i in range(a, b)]
    traza.append(f"Paso 2: Correspondencias en segmento de corte: {', '.join(mapeo_str)}")

    # 2. Mapear elementos del segundo padre no presentes en el primer padre
    for i in range(a, b):
        val_b = padre_b[i]
        if val_b not in padre_a[a:b]:
            pos = i
            while a <= pos < b:
                val_a = padre_a[pos]
                pos = padre_b.index(val_a)
            hijo[pos] = val_b
            traza.append(f"        Elemento {val_b} de {nombre_b} se mapea a la posición libre {pos}.")

    traza.append(f"        Estado del hijo tras mapeo: {hijo}")

    # 3. Llenar los vacíos restantes directamente desde el segundo padre
    for i in range(n):
        if hijo[i] is None:
            hijo[i] = padre_b[i]
            traza.append(f"        Posición vacía {i} se copia directo de {nombre_b}[{i}] = {padre_b[i]}")

    traza.append(f"Paso 3: Hijo final resultante: {hijo}")
    return hijo, traza


if __name__ == "__main__":
    p1 = [1, 2, 3, 4, 5, 6, 7, 8]
    p2 = [8, 5, 2, 1, 3, 6, 4, 7]
    a, b = 3, 6  # Posiciones 3 a 5 (índices 3, 4, 5: genes 4, 5, 6 de p1)

    print("=" * 65)
    print("DEMOSTRACIÓN PASO A PASO: CRUZA OX (8 GENES, CORTE [3:6])")
    print("=" * 65)
    print(f"Padre 1: {p1}")
    print(f"Padre 2: {p2}")
    print(f"Segmento p1[3:6] = {p1[3:6]}, Segmento p2[3:6] = {p2[3:6]}\n")

    print("--- Generando Hijo 1 (OX) [Segmento copiado de p1, resto de p2] ---")
    hijo1_ox, traza_ox1 = cruza_ox_paso_a_paso(p1, p2, a, b, nombre_a="p1", nombre_b="p2")
    for linea in traza_ox1:
        print(linea)
    print(f"¿Hijo 1 es permutación válida?: {es_valido(hijo1_ox, 8)}\n")

    print("--- Generando Hijo 2 (OX) [Segmento copiado de p2, resto de p1] ---")
    hijo2_ox, traza_ox2 = cruza_ox_paso_a_paso(p2, p1, a, b, nombre_a="p2", nombre_b="p1")
    for linea in traza_ox2:
        print(linea)
    print(f"¿Hijo 2 es permutación válida?: {es_valido(hijo2_ox, 8)}\n")

    print("=" * 65)
    print("DEMOSTRACIÓN PASO A PASO: CRUZA PMX (8 GENES, CORTE [3:6])")
    print("=" * 65)
    print("--- Generando Hijo 1 (PMX) [Segmento copiado de p1, resto de p2] ---")
    hijo1_pmx, traza_pmx1 = cruza_pmx_paso_a_paso(p1, p2, a, b, nombre_a="p1", nombre_b="p2")
    for linea in traza_pmx1:
        print(linea)
    print(f"¿Hijo 1 es permutación válida?: {es_valido(hijo1_pmx, 8)}\n")

    print("--- Generando Hijo 2 (PMX) [Segmento copiado de p2, resto de p1] ---")
    hijo2_pmx, traza_pmx2 = cruza_pmx_paso_a_paso(p2, p1, a, b, nombre_a="p2", nombre_b="p1")
    for linea in traza_pmx2:
        print(linea)
    print(f"¿Hijo 2 es permutación válida?: {es_valido(hijo2_pmx, 8)}\n")

    print("=" * 65)
    print("RESUMEN COMPARATIVO DE HIJOS GENERADOS:")
    print("=" * 65)
    print(f"Hijo 1 (OX):  {hijo1_ox}")
    print(f"Hijo 1 (PMX): {hijo1_pmx}")
    print(f"¿Hijo 1 OX != PMX?: {hijo1_ox != hijo1_pmx}")
    print(f"Hijo 2 (OX):  {hijo2_ox}")
    print(f"Hijo 2 (PMX): {hijo2_pmx}")
    print(f"¿Hijo 2 OX != PMX?: {hijo2_ox != hijo2_pmx}")

    assert hijo1_ox  == [2, 1, 3, 4, 5, 6, 7, 8]
    assert hijo2_ox  == [2, 4, 5, 1, 3, 6, 7, 8]
    assert hijo1_pmx == [8, 3, 2, 4, 5, 6, 1, 7]
    assert hijo2_pmx == [4, 2, 5, 1, 3, 6, 7, 8]
    print("Todas las pruebas pasaron.")

