"""
=============================================================
  EVENTOESPACIO — Motor del Algoritmo Genético  (actualizado)
  Cambios respecto a la versión original:
    · ejecutar_ag acepta rutas de CSV externas (datasets cargados)
    · algoritmo_genetico registra hist_peor en cada generación
    · algoritmo_genetico mantiene top3 mejores individuos
    · ejecutar_ag devuelve hist_peor y top3 al frontend
=============================================================
"""

import csv
import random
import math
import copy
import os

# ─────────────────────────────────────────────
# RUTAS DE ARCHIVOS POR DEFECTO
# ─────────────────────────────────────────────
BASE_DIR              = os.path.dirname(os.path.abspath(__file__))
DATA_DIR              = os.path.join(BASE_DIR, "data")
ARCHIVO_ELEMENTOS     = os.path.join(DATA_DIR, "catalogo_elementos.csv")
ARCHIVO_RESTRICCIONES = os.path.join(DATA_DIR, "catalogo_restricciones.csv")

# ─────────────────────────────────────────────
# 1. PARÁMETROS POR DEFECTO
# ─────────────────────────────────────────────
TAM_POBLACION  = 10
P_CRUZA        = 0.70
P_MUT_IND      = 0.30
P_MUT_GEN      = 0.25
N_GENERACIONES = 200

ANCHO_GRID = 20
ALTO_GRID  = 20


# ─────────────────────────────────────────────
# 2. LECTURA DE BASE DE CONOCIMIENTO (CSV)
# ─────────────────────────────────────────────
def cargar_elementos(archivo=None):
    if archivo is None:
        archivo = ARCHIVO_ELEMENTOS
    elementos = []
    with open(archivo, newline='', encoding='utf-8') as f:
        for fila in csv.DictReader(f):
            elementos.append({
                "id":              int(fila["id"]),
                "tipo":            fila["tipo"],
                "prioridad":       fila["prioridad"],
                "ancho":           int(fila["ancho"]),
                "alto":            int(fila["alto"]),
                "requiere_acceso": int(fila["requiere_acceso"])
            })
    return elementos


def cargar_restricciones(archivo=None):
    if archivo is None:
        archivo = ARCHIVO_RESTRICCIONES
    restricciones = []
    with open(archivo, newline='', encoding='utf-8') as f:
        for fila in csv.DictReader(f):
            restricciones.append({
                "id":          int(fila["id"]),
                "tipo":        fila["tipo"],
                "x":           int(fila["x"]),
                "y":           int(fila["y"]),
                "descripcion": fila["descripcion"]
            })
    return restricciones


def obtener_entradas(restricciones):
    return [r for r in restricciones if r["tipo"] == "entrada"]


def obtener_celdas_restringidas(restricciones):
    return {(r["x"], r["y"]) for r in restricciones if r["tipo"] == "zona_restringida"}


# ─────────────────────────────────────────────
# 3. UTILIDADES GEOMÉTRICAS
# ─────────────────────────────────────────────
def celdas_del_elemento(x, y, ancho, alto):
    return {(x + dx, y + dy)
            for dx in range(ancho)
            for dy in range(alto)}


def elemento_es_valido(x, y, ancho, alto, celdas_restringidas):
    return celdas_del_elemento(x, y, ancho, alto).isdisjoint(celdas_restringidas)


def se_solapan(x1, y1, w1, h1, x2, y2, w2, h2):
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1)


def hay_solapamiento(individuo, elementos):
    n = len(individuo)
    for i in range(n):
        for j in range(i + 1, n):
            x1, y1 = individuo[i]; e1 = elementos[i]
            x2, y2 = individuo[j]; e2 = elementos[j]
            if se_solapan(x1, y1, e1["ancho"], e1["alto"],
                          x2, y2, e2["ancho"], e2["alto"]):
                return True
    return False


# ─────────────────────────────────────────────
# 4. REPRESENTACIÓN DEL INDIVIDUO
# ─────────────────────────────────────────────
def crear_individuo(elementos, celdas_restringidas, W=None, H=None):
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    individuo = []
    for elem in elementos:
        intentos = 0
        while True:
            intentos += 1
            x = random.randint(0, max(0, W - elem["ancho"]))
            y = random.randint(0, max(0, H - elem["alto"]))
            if elemento_es_valido(x, y, elem["ancho"], elem["alto"], celdas_restringidas):
                break
            if intentos > 500:
                break
        individuo.append((x, y))
    return individuo


def crear_poblacion(elementos, celdas_restringidas, tam=None, W=None, H=None):
    if tam is None: tam = TAM_POBLACION
    return [crear_individuo(elementos, celdas_restringidas, W, H)
            for _ in range(tam)]


# ─────────────────────────────────────────────
# 5. FUNCIONES OBJETIVO
# ─────────────────────────────────────────────
def calcular_O1_distribucion(individuo, W=None, H=None):
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    mitad_x = W / 2
    mitad_y = H / 2
    cuadrantes = [0, 0, 0, 0]

    for (x, y) in individuo:
        if   x >= mitad_x and y >= mitad_y: cuadrantes[0] += 1
        elif x <  mitad_x and y >= mitad_y: cuadrantes[1] += 1
        elif x >= mitad_x and y <  mitad_y: cuadrantes[2] += 1
        else:                               cuadrantes[3] += 1

    total = len(individuo)
    if total == 0:
        return 0
    promedio = total / 4
    O1 = 1 - ((promedio - min(cuadrantes)) / total)
    return max(0.0, min(1.0, O1))


def calcular_O2_flujo(individuo, elementos, W=None, H=None):
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    celdas_totales  = W * H
    celdas_ocupadas = sum(e["ancho"] * e["alto"] for e in elementos)
    O2 = (celdas_totales - celdas_ocupadas) / celdas_totales
    return max(0.0, min(1.0, O2))


def calcular_O3_conectividad(individuo, elementos, entradas, W=None, H=None):
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    if not entradas:
        return 1.0

    distancia_max    = math.sqrt(W**2 + H**2)
    elementos_acceso = [i for i, e in enumerate(elementos) if e["requiere_acceso"] == 1]

    if not elementos_acceso:
        return 1.0

    suma = 0
    for i in elementos_acceso:
        x, y = individuo[i]
        suma += min(math.sqrt((x - en["x"])**2 + (y - en["y"])**2)
                    for en in entradas)

    promedio_dist = suma / len(elementos_acceso)
    O3 = 1 - (promedio_dist / distancia_max)
    return max(0.0, min(1.0, O3))


def calcular_O4_prioridad(individuo, elementos, W=None, H=None):
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    cx = W / 2
    cy = H / 2
    radio = min(W, H) / 3

    alta = [i for i, e in enumerate(elementos) if e["prioridad"] == "alta"]
    if not alta:
        return 1.0

    en_zona = sum(
        1 for i in alta
        if math.sqrt((individuo[i][0] - cx)**2 + (individuo[i][1] - cy)**2) <= radio
    )
    return max(0.0, min(1.0, en_zona / len(alta)))


def calcular_aptitud(individuo, elementos, entradas, W=None, H=None):
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    if hay_solapamiento(individuo, elementos):
        return 0.0

    O1 = calcular_O1_distribucion(individuo, W, H)
    O2 = calcular_O2_flujo(individuo, elementos, W, H)
    O3 = calcular_O3_conectividad(individuo, elementos, entradas, W, H)
    O4 = calcular_O4_prioridad(individuo, elementos, W, H)

    return O1 * O2 * O3 * O4


# ─────────────────────────────────────────────
# 6. SELECCIÓN POR TORNEO
# ─────────────────────────────────────────────
def seleccion_torneo(poblacion, aptitudes, k=3):
    candidatos = random.sample(range(len(poblacion)), min(k, len(poblacion)))
    mejor_idx  = max(candidatos, key=lambda i: aptitudes[i])
    return copy.deepcopy(poblacion[mejor_idx])


# ─────────────────────────────────────────────
# 7. CRUZAMIENTO
# ─────────────────────────────────────────────
def cruzamiento(padre1, padre2, p_cruza=None):
    if p_cruza is None: p_cruza = P_CRUZA
    if random.random() < p_cruza:
        punto = random.randint(1, len(padre1) - 1)
        return padre1[:punto] + padre2[punto:]
    return copy.deepcopy(padre1)


# ─────────────────────────────────────────────
# 8. MUTACIÓN
# ─────────────────────────────────────────────
def mutacion(individuo, elementos, celdas_restringidas, pmi=None, pmg=None, W=None, H=None):
    if pmi is None: pmi = P_MUT_IND
    if pmg is None: pmg = P_MUT_GEN
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    if random.random() < pmi:
        for i in range(len(individuo)):
            if random.random() < pmg:
                intentos = 0
                while True:
                    intentos += 1
                    x = random.randint(0, max(0, W - elementos[i]["ancho"]))
                    y = random.randint(0, max(0, H - elementos[i]["alto"]))
                    if elemento_es_valido(x, y, elementos[i]["ancho"],
                                         elementos[i]["alto"], celdas_restringidas):
                        break
                    if intentos > 500:
                        break
                individuo[i] = (x, y)
    return individuo


# ─────────────────────────────────────────────
# 9. ALGORITMO GENÉTICO PRINCIPAL
#    NUEVO: registra hist_peor y mantiene top3
# ─────────────────────────────────────────────
def _insertar_top3(top3, individuo, aptitud):
    """
    Mantiene una lista de los 3 mejores individuos únicos (sin duplicados exactos).
    top3 = lista de dicts {'individuo': [...], 'aptitud': float}
    """
    # ¿Ya existe uno con aptitud muy cercana? (tolerancia 1e-6)
    for entry in top3:
        if abs(entry["aptitud"] - aptitud) < 1e-6:
            return  # considerado duplicado

    top3.append({"individuo": copy.deepcopy(individuo), "aptitud": aptitud})
    top3.sort(key=lambda e: e["aptitud"], reverse=True)
    if len(top3) > 3:
        top3.pop()


def algoritmo_genetico(elementos, entradas, celdas_restringidas, params=None):
    """
    Ejecuta el AG completo.
    Retorna:
      mejor_individuo, mejor_aptitud,
      historial_mejor, historial_promedio, historial_peor,
      top3
    """
    tam_pob = params.get("tam_poblacion", TAM_POBLACION) if params else TAM_POBLACION
    pc      = params.get("p_cruza",       P_CRUZA)       if params else P_CRUZA
    pmi     = params.get("p_mut_ind",     P_MUT_IND)     if params else P_MUT_IND
    pmg     = params.get("p_mut_gen",     P_MUT_GEN)     if params else P_MUT_GEN
    n_gen   = params.get("generaciones",  N_GENERACIONES) if params else N_GENERACIONES
    W       = params.get("ancho",         ANCHO_GRID)    if params else ANCHO_GRID
    H       = params.get("alto",          ALTO_GRID)     if params else ALTO_GRID

    poblacion = crear_poblacion(elementos, celdas_restringidas, tam_pob, W, H)

    historial_mejor    = []
    historial_promedio = []
    historial_peor     = []          # ← NUEVO

    mejor_individuo = None
    mejor_aptitud   = -1.0
    top3            = []             # ← NUEVO

    for generacion in range(n_gen):

        aptitudes    = [calcular_aptitud(ind, elementos, entradas, W, H) for ind in poblacion]
        mejor_gen    = max(aptitudes)
        peor_gen     = min(aptitudes)
        promedio_gen = sum(aptitudes) / len(aptitudes)

        historial_mejor.append(round(mejor_gen,    6))
        historial_promedio.append(round(promedio_gen, 6))
        historial_peor.append(round(peor_gen,      6))   # ← NUEVO

        idx_mejor = aptitudes.index(mejor_gen)
        if mejor_gen > mejor_aptitud:
            mejor_aptitud   = mejor_gen
            mejor_individuo = copy.deepcopy(poblacion[idx_mejor])

        # Actualiza top3 con todos los de esta generación
        for ind, apt in zip(poblacion, aptitudes):
            _insertar_top3(top3, ind, apt)

        # Nueva generación con elitismo estricto
        nueva_poblacion = [copy.deepcopy(mejor_individuo)]

        while len(nueva_poblacion) < tam_pob:
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            hijo   = cruzamiento(padre1, padre2, pc)
            hijo   = mutacion(hijo, elementos, celdas_restringidas, pmi, pmg, W, H)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    return (mejor_individuo, mejor_aptitud,
            historial_mejor, historial_promedio, historial_peor,
            top3)


# ─────────────────────────────────────────────
# 10. FUNCIÓN PÚBLICA PARA app.py
# ─────────────────────────────────────────────
def _individuo_a_tabla(individuo, elementos):
    tabla = []
    for i, e in enumerate(elementos):
        x, y = individuo[i]
        tabla.append({
            "id":        e["id"],
            "tipo":      e["tipo"],
            "prioridad": e["prioridad"],
            "x": x, "y": y,
            "ancho": e["ancho"],
            "alto":  e["alto"]
        })
    return tabla


def ejecutar_ag(params):
    """
    Punto de entrada único que usa app.py.
    Acepta 'archivo_elementos' y 'archivo_restricciones' en params
    para soportar datasets cargados desde el frontend.
    """
    W = params.get("ancho", ANCHO_GRID)
    H = params.get("alto",  ALTO_GRID)

    # Soporte de dataset externo
    arch_elem = params.get("archivo_elementos")
    arch_rest = params.get("archivo_restricciones")

    elementos     = cargar_elementos(arch_elem)
    restricciones = cargar_restricciones(arch_rest)
    entradas      = obtener_entradas(restricciones)
    celdas_rest   = obtener_celdas_restringidas(restricciones)

    (mejor_ind, mejor_apt,
     hist_mejor, hist_promedio, hist_peor,
     top3) = algoritmo_genetico(elementos, entradas, celdas_rest, params)

    O1 = round(calcular_O1_distribucion(mejor_ind, W, H), 4)
    O2 = round(calcular_O2_flujo(mejor_ind, elementos, W, H), 4)
    O3 = round(calcular_O3_conectividad(mejor_ind, elementos, entradas, W, H), 4)
    O4 = round(calcular_O4_prioridad(mejor_ind, elementos, W, H), 4)

    # Serializa top3
    top3_serial = []
    for rank, entry in enumerate(top3, 1):
        top3_serial.append({
            "rank":    rank,
            "aptitud": round(entry["aptitud"], 6),
            "tabla":   _individuo_a_tabla(entry["individuo"], elementos)
        })

    return {
        "aptitud":        round(mejor_apt, 6),
        "O1": O1, "O2": O2, "O3": O3, "O4": O4,
        "hist_mejor":     hist_mejor,
        "hist_promedio":  hist_promedio,
        "hist_peor":      hist_peor,          # ← NUEVO
        "top3":           top3_serial,        # ← NUEVO
        "tabla":          _individuo_a_tabla(mejor_ind, elementos),
        "restricciones":  restricciones,
        "ancho": W, "alto": H
    }


# ─────────────────────────────────────────────
# 11. PUNTO DE ENTRADA DIRECTO
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    elementos     = cargar_elementos()
    restricciones = cargar_restricciones()
    entradas      = obtener_entradas(restricciones)
    celdas_rest   = obtener_celdas_restringidas(restricciones)

    print(f"Elementos cargados   : {len(elementos)}")
    print(f"Restricciones        : {len(restricciones)}")

    (mejor_ind, mejor_apt,
     hist_mejor, hist_prom, hist_peor,
     top3) = algoritmo_genetico(elementos, entradas, celdas_rest)

    print(f"\n★ APTITUD FINAL = {mejor_apt:.6f}")
    print("\nTOP 3:")
    for e in top3:
        print(f"  #{e['rank']} → {e['aptitud']:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(hist_mejor, color='#a855f7', linewidth=2,   label='Mejor aptitud')
    plt.plot(hist_prom,  color='#ec4899', linewidth=1.5, linestyle='--', label='Promedio')
    plt.plot(hist_peor,  color='#f97316', linewidth=1.5, linestyle=':',  label='Peor aptitud')
    plt.title("EVENTOESPACIO — Evolución del Fitness")
    plt.xlabel("Generación"); plt.ylabel("Aptitud (0-1)"); plt.ylim(0, 1)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("evolucion_fitness.png", dpi=150); plt.show()