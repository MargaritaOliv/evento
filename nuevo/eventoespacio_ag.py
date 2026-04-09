
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
TAM_POBLACION  = 100
P_CRUZA        = 0.70
P_MUT_IND      = 0.30
P_MUT_GEN      = 0.25
N_GENERACIONES = 400

ANCHO_GRID = 20
ALTO_GRID  = 20

# Pesos de la función fitness (deben sumar 1.0)
W_DISTRIBUCION  = 0.25   # WE
W_FLUJO         = 0.30   # WF
W_CONECTIVIDAD  = 0.30   # WC
W_PRIORIDAD     = 0.15   # WP


# ─────────────────────────────────────────────
# 2. LECTURA DE BASE DE CONOCIMIENTO (CSV)
# ─────────────────────────────────────────────
def cargar_elementos(archivo=None):
    if archivo is None:
        archivo = ARCHIVO_ELEMENTOS
    elementos = []
    with open(archivo, newline='', encoding='utf-8') as f:
        for fila in csv.DictReader(f):
            # Soporte de ambos formatos de CSV (antiguo y nuevo)
            req = fila.get("requiere_acceso") or fila.get("requires_access", "0")
            if isinstance(req, str):
                req = 1 if req.strip().lower() in ("1", "true") else 0
            elementos.append({
                "id":              int(fila["id"]),
                "nombre":          fila.get("nombre", fila.get("tipo", "")),
                "tipo":            fila["tipo"],
                "prioridad":       fila["prioridad"],
                "ancho":           int(fila["ancho"]),
                "alto":            int(fila["alto"]),
                "requiere_acceso": int(req)
            })
    return elementos


def cargar_restricciones(archivo=None):
    if archivo is None:
        archivo = ARCHIVO_RESTRICCIONES
    restricciones = []
    with open(archivo, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for fila in reader:
            # Soporte de ambos formatos: (x,y) y (x1,y1,x2,y2)
            if "x1" in fila:
                # Formato nuevo: usamos el punto medio como referencia
                x = (int(fila["x1"]) + int(fila["x2"])) // 2
                y = (int(fila["y1"]) + int(fila["y2"])) // 2
                # Guardamos también x1,y1,x2,y2 para celdas restringidas
                x1, y1 = int(fila["x1"]), int(fila["y1"])
                x2, y2 = int(fila["x2"]), int(fila["y2"])
            else:
                x = int(fila["x"])
                y = int(fila["y"])
                x1 = y1 = x2 = y2 = None

            restricciones.append({
                "id":          int(fila["id"]),
                "tipo":        fila["tipo"],
                "x":           x,
                "y":           y,
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "descripcion": fila.get("descripcion", "")
            })
    return restricciones


def obtener_entradas(restricciones):
    return [r for r in restricciones if r["tipo"] == "entrada"]


# FIX 3: función separada para salidas de emergencia
def obtener_salidas(restricciones):
    return [r for r in restricciones if r["tipo"] == "salida"]


def obtener_celdas_restringidas(restricciones):
    celdas = set()
    for r in restricciones:
        if r["tipo"] in ("zona_restringida", "columna"):
            if r["x1"] is not None:
                for cx in range(r["x1"], r["x2"]):
                    for cy in range(r["y1"], r["y2"]):
                        celdas.add((cx, cy))
            else:
                celdas.add((r["x"], r["y"]))
    return celdas


# ─────────────────────────────────────────────
# 3. UTILIDADES GEOMÉTRICAS
# ─────────────────────────────────────────────
def dims_efectivas(elem, rotado):
    """Devuelve (ancho, alto) aplicando rotación."""
    if rotado:
        return elem["alto"], elem["ancho"]
    return elem["ancho"], elem["alto"]


def celdas_del_elemento(x, y, ancho, alto):
    return {(x + dx, y + dy)
            for dx in range(ancho)
            for dy in range(alto)}


def elemento_es_valido(x, y, ancho, alto, celdas_restringidas):
    return celdas_del_elemento(x, y, ancho, alto).isdisjoint(celdas_restringidas)


def se_solapan(x1, y1, w1, h1, x2, y2, w2, h2):
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1)


def area_solapamiento(x1, y1, w1, h1, x2, y2, w2, h2):
    """Área de intersección entre dos rectángulos (0 si no se tocan)."""
    dx = min(x1 + w1, x2 + w2) - max(x1, x2)
    dy = min(y1 + h1, y2 + h2) - max(y1, y2)
    if dx > 0 and dy > 0:
        return dx * dy
    return 0


# ─────────────────────────────────────────────
# FIX 1 — Factor de Superposición suave
# ─────────────────────────────────────────────
def calcular_factor_superposicion(individuo, elementos, W, H):
    """
    Retorna un valor ∈ (0, 1].
    · 1.0  → sin solapamiento alguno (solución perfectamente viable)
    · ~0   → solapamiento masivo (fuerte penalización, pero el individuo
              sigue siendo "comparable" con otros muy malos)

    Fórmula:
        area_encimada_total / area_total_elementos
        factor = 1 - clamp(ratio, 0, 1)
    Se añade un piso de 0.05 para que nunca sea exactamente 0
    (el algoritmo siempre puede distinguir "más malo" de "menos malo").
    """
    n = len(individuo)
    area_encimada = 0
    area_total = sum(dims_efectivas(elementos[i], individuo[i][2])[0] *
                     dims_efectivas(elementos[i], individuo[i][2])[1]
                     for i in range(n))

    for i in range(n):
        xi, yi, ri = individuo[i]
        wi, hi = dims_efectivas(elementos[i], ri)
        for j in range(i + 1, n):
            xj, yj, rj = individuo[j]
            wj, hj = dims_efectivas(elementos[j], rj)
            area_encimada += area_solapamiento(xi, yi, wi, hi, xj, yj, wj, hj)

    if area_total == 0:
        return 1.0

    ratio = area_encimada / area_total
    factor = 1.0 - min(ratio, 1.0)
    return max(0.05, factor)   # piso de 0.05 para mantener gradiente


# ─────────────────────────────────────────────
# 4. REPRESENTACIÓN DEL INDIVIDUO  (FIX 4: rotación)
# ─────────────────────────────────────────────
# Cada gen = (x, y, rotado)  con rotado ∈ {0, 1}

def crear_individuo(elementos, celdas_restringidas, W=None, H=None):
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    individuo = []
    for elem in elementos:
        rotado = random.randint(0, 1)
        ancho, alto = dims_efectivas(elem, rotado)

        # Asegura que el elemento cabe; si no, no rota
        if ancho > W or alto > H:
            rotado = 0
            ancho, alto = dims_efectivas(elem, rotado)

        intentos = 0
        while True:
            intentos += 1
            x = random.randint(0, max(0, W - ancho))
            y = random.randint(0, max(0, H - alto))
            if elemento_es_valido(x, y, ancho, alto, celdas_restringidas):
                break
            if intentos > 500:
                break
        individuo.append((x, y, rotado))
    return individuo


def crear_poblacion(elementos, celdas_restringidas, tam=None, W=None, H=None):
    if tam is None: tam = TAM_POBLACION
    return [crear_individuo(elementos, celdas_restringidas, W, H)
            for _ in range(tam)]


# ─────────────────────────────────────────────
# 5. FUNCIONES OBJETIVO
# ─────────────────────────────────────────────

def calcular_O1_distribucion(individuo, elementos, W=None, H=None):
    """
    O1: Equilibrio de distribución por cuadrantes.
    Mide qué tan uniformemente están repartidos los elementos.
    """
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    mitad_x = W / 2
    mitad_y = H / 2
    cuadrantes = [0, 0, 0, 0]

    for (x, y, _) in individuo:
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


# FIX 2 — Flujo basado en separación real entre elementos
def calcular_O2_flujo(individuo, elementos, W=None, H=None):
    """
    O2: Calidad del flujo de personas.

    Mide la separación mínima promedio entre pares de elementos
    vecinos (distancia borde-a-borde), normalizada por la
    diagonal del recinto.

    Cuanto mayor sea la separación entre elementos, más espacio
    libre existe para que la gente circule → menos "muros".

    Para n elementos, se calculan todas las separaciones par a par
    y se promedian.  Se normaliza con la diagonal del recinto.
    """
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    n = len(individuo)
    if n < 2:
        return 1.0

    diagonal = math.sqrt(W**2 + H**2)
    separaciones = []

    for i in range(n):
        xi, yi, ri = individuo[i]
        wi, hi = dims_efectivas(elementos[i], ri)
        for j in range(i + 1, n):
            xj, yj, rj = individuo[j]
            wj, hj = dims_efectivas(elementos[j], rj)

            # Distancia borde-a-borde en X e Y
            sep_x = max(0, max(xi, xj) - min(xi + wi, xj + wj))
            sep_y = max(0, max(yi, yj) - min(yi + hi, yj + hj))

            # Distancia euclidiana borde-a-borde
            sep = math.sqrt(sep_x**2 + sep_y**2)
            separaciones.append(sep)

    if not separaciones:
        return 1.0

    sep_promedio = sum(separaciones) / len(separaciones)
    # Normalizar: separación ideal = ~10 % de la diagonal
    # Saturamos en 1.0 para no penalizar "demasiado espacio"
    O2 = min(1.0, sep_promedio / (diagonal * 0.15))
    return max(0.0, O2)


# FIX 3 — Conectividad: entradas (40 %) + salidas de emergencia (60 %)
def calcular_O3_conectividad(individuo, elementos, entradas, salidas, W=None, H=None):
    """
    O3: Accesibilidad ponderada.

    · 40 % → proximidad a entradas  (experiencia del asistente)
    · 60 % → proximidad a salidas de emergencia  (protección civil)

    Solo se evalúan elementos con requiere_acceso == 1.
    Si no hay entradas ni salidas definidas, retorna 1.0.
    """
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    distancia_max = math.sqrt(W**2 + H**2)
    elementos_acceso = [i for i, e in enumerate(elementos)
                        if e["requiere_acceso"] == 1]

    if not elementos_acceso:
        return 1.0

    def score_proximity(idx_lista, puntos):
        """Promedio de (1 - dist_normalizada) para los elementos de la lista."""
        if not puntos:
            return 1.0
        total = 0.0
        for i in idx_lista:
            x, y, _ = individuo[i]
            d = min(math.sqrt((x - p["x"])**2 + (y - p["y"])**2)
                    for p in puntos)
            total += 1.0 - (d / distancia_max)
        return total / len(idx_lista)

    score_entradas = score_proximity(elementos_acceso, entradas)
    score_salidas  = score_proximity(elementos_acceso, salidas) if salidas else 1.0

    O3 = 0.40 * score_entradas + 0.60 * score_salidas
    return max(0.0, min(1.0, O3))


def calcular_O4_prioridad(individuo, elementos, W=None, H=None):
    """
    O4: Elementos de alta prioridad cerca del centro del recinto.
    """
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    cx = W / 2
    cy = H / 2
    radio = min(W, H) / 3

    # Soporte de prioridad numérica (nuevo CSV) y textual (antiguo CSV)
    def es_alta(e):
        p = e["prioridad"]
        try:
            return float(p) >= 4
        except (ValueError, TypeError):
            return str(p).strip().lower() == "alta"

    alta = [i for i, e in enumerate(elementos) if es_alta(e)]
    if not alta:
        return 1.0

    en_zona = sum(
        1 for i in alta
        if math.sqrt((individuo[i][0] - cx)**2 + (individuo[i][1] - cy)**2) <= radio
    )
    return max(0.0, min(1.0, en_zona / len(alta)))


# ─────────────────────────────────────────────
# FITNESS — Suma Ponderada × Factor_Superposicion
# ─────────────────────────────────────────────
def calcular_aptitud(individuo, elementos, entradas, salidas, W=None, H=None,
                     WE=W_DISTRIBUCION, WF=W_FLUJO, WC=W_CONECTIVIDAD, WP=W_PRIORIDAD):
    """
    Aptitud = [(O1·WE) + (O2·WF) + (O3·WC) + (O4·WP)] × Factor_Superposicion

    FIX 1: Factor_Superposicion es continuo (0.05 – 1.0), no binario.
           El algoritmo siempre puede distinguir soluciones "más malas"
           de "menos malas", aunque todas se encimen.
    """
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    # Suma ponderada de objetivos
    O1 = calcular_O1_distribucion(individuo, elementos, W, H)
    O2 = calcular_O2_flujo(individuo, elementos, W, H)
    O3 = calcular_O3_conectividad(individuo, elementos, entradas, salidas, W, H)
    O4 = calcular_O4_prioridad(individuo, elementos, W, H)

    suma_ponderada = (O1 * WE) + (O2 * WF) + (O3 * WC) + (O4 * WP)

    # FIX 1: penalización suave por área encimada
    factor_sup = calcular_factor_superposicion(individuo, elementos, W, H)

    aptitud = suma_ponderada * factor_sup
    return max(0.0, min(1.0, aptitud))


# ─────────────────────────────────────────────
# 6. SELECCIÓN POR TORNEO
# ─────────────────────────────────────────────
def seleccion_torneo(poblacion, aptitudes, k=3):
    candidatos = random.sample(range(len(poblacion)), min(k, len(poblacion)))
    mejor_idx  = max(candidatos, key=lambda i: aptitudes[i])
    return copy.deepcopy(poblacion[mejor_idx])


# ─────────────────────────────────────────────
# 7. CRUZAMIENTO  (actualizado para terna x,y,rotado)
# ─────────────────────────────────────────────
def cruzamiento(padre1, padre2, p_cruza=None):
    if p_cruza is None: p_cruza = P_CRUZA
    if random.random() < p_cruza:
        punto = random.randint(1, len(padre1) - 1)
        return padre1[:punto] + padre2[punto:]
    return copy.deepcopy(padre1)


# ─────────────────────────────────────────────
# 8. MUTACIÓN  (FIX 4: puede mutar también la rotación)
# ─────────────────────────────────────────────
def mutacion(individuo, elementos, celdas_restringidas, pmi=None, pmg=None, W=None, H=None):
    if pmi is None: pmi = P_MUT_IND
    if pmg is None: pmg = P_MUT_GEN
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    if random.random() < pmi:
        for i in range(len(individuo)):
            if random.random() < pmg:
                x_old, y_old, rot_old = individuo[i]

                # 30 % de probabilidad de mutar rotación
                rotado = 1 - rot_old if random.random() < 0.3 else rot_old
                ancho, alto = dims_efectivas(elementos[i], rotado)

                # Si con la nueva rotación no cabe, revertir
                if ancho > W or alto > H:
                    rotado = rot_old
                    ancho, alto = dims_efectivas(elementos[i], rotado)

                intentos = 0
                while True:
                    intentos += 1
                    x = random.randint(0, max(0, W - ancho))
                    y = random.randint(0, max(0, H - alto))
                    if elemento_es_valido(x, y, ancho, alto, celdas_restringidas):
                        break
                    if intentos > 500:
                        break
                individuo[i] = (x, y, rotado)
    return individuo


# ─────────────────────────────────────────────
# 9. ALGORITMO GENÉTICO PRINCIPAL
# ─────────────────────────────────────────────
def _insertar_top3(top3, individuo, aptitud):
    for entry in top3:
        if abs(entry["aptitud"] - aptitud) < 1e-6:
            return
    top3.append({"individuo": copy.deepcopy(individuo), "aptitud": aptitud})
    top3.sort(key=lambda e: e["aptitud"], reverse=True)
    if len(top3) > 3:
        top3.pop()


def algoritmo_genetico(elementos, entradas, salidas, celdas_restringidas, params=None):
    """
    Ejecuta el AG completo con los 4 fixes.
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
    historial_peor     = []

    mejor_individuo = None
    mejor_aptitud   = -1.0
    top3            = []

    for _ in range(n_gen):
        aptitudes = [
            calcular_aptitud(ind, elementos, entradas, salidas, W, H)
            for ind in poblacion
        ]

        mejor_gen    = max(aptitudes)
        peor_gen     = min(aptitudes)
        promedio_gen = sum(aptitudes) / len(aptitudes)

        historial_mejor.append(round(mejor_gen,    6))
        historial_promedio.append(round(promedio_gen, 6))
        historial_peor.append(round(peor_gen,      6))

        idx_mejor = aptitudes.index(mejor_gen)
        if mejor_gen > mejor_aptitud:
            mejor_aptitud   = mejor_gen
            mejor_individuo = copy.deepcopy(poblacion[idx_mejor])

        for ind, apt in zip(poblacion, aptitudes):
            _insertar_top3(top3, ind, apt)

        # Elitismo: el mejor pasa directo
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
        x, y, rotado = individuo[i]
        ancho, alto = dims_efectivas(e, rotado)
        tabla.append({
            "id":        e["id"],
            "nombre":    e.get("nombre", e["tipo"]),
            "tipo":      e["tipo"],
            "prioridad": e["prioridad"],
            "x": x, "y": y,
            "ancho": ancho,
            "alto":  alto,
            "rotado": bool(rotado)        # FIX 4: se expone al frontend
        })
    return tabla


def ejecutar_ag(params):
    """
    Punto de entrada único para app.py.
    Acepta 'archivo_elementos' y 'archivo_restricciones' en params.
    """
    W = params.get("ancho", ANCHO_GRID)
    H = params.get("alto",  ALTO_GRID)

    arch_elem = params.get("archivo_elementos")
    arch_rest = params.get("archivo_restricciones")

    elementos     = cargar_elementos(arch_elem)
    restricciones = cargar_restricciones(arch_rest)
    entradas      = obtener_entradas(restricciones)
    salidas       = obtener_salidas(restricciones)          # FIX 3
    celdas_rest   = obtener_celdas_restringidas(restricciones)

    (mejor_ind, mejor_apt,
     hist_mejor, hist_promedio, hist_peor,
     top3) = algoritmo_genetico(elementos, entradas, salidas, celdas_rest, params)

    # Descomposición del fitness del mejor individuo
    O1 = round(calcular_O1_distribucion(mejor_ind, elementos, W, H), 4)
    O2 = round(calcular_O2_flujo(mejor_ind, elementos, W, H), 4)
    O3 = round(calcular_O3_conectividad(mejor_ind, elementos, entradas, salidas, W, H), 4)
    O4 = round(calcular_O4_prioridad(mejor_ind, elementos, W, H), 4)
    FS = round(calcular_factor_superposicion(mejor_ind, elementos, W, H), 4)

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
        "factor_superposicion": FS,            # FIX 1: visible en frontend
        "hist_mejor":     hist_mejor,
        "hist_promedio":  hist_promedio,
        "hist_peor":      hist_peor,
        "top3":           top3_serial,
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
    salidas       = obtener_salidas(restricciones)
    celdas_rest   = obtener_celdas_restringidas(restricciones)

    print(f"Elementos cargados   : {len(elementos)}")
    print(f"Restricciones        : {len(restricciones)}")
    print(f"Entradas             : {len(entradas)}")
    print(f"Salidas emergencia   : {len(salidas)}")

    (mejor_ind, mejor_apt,
     hist_mejor, hist_prom, hist_peor,
     top3) = algoritmo_genetico(elementos, entradas, salidas, celdas_rest)

    print(f"\n★ APTITUD FINAL = {mejor_apt:.6f}")
    print("\nTOP 3:")
    for e in top3:
        print(f"  #{e['rank']} → {e['aptitud']:.6f}")

    plt.figure(figsize=(10, 5))
    plt.plot(hist_mejor, color='#a855f7', linewidth=2,   label='Mejor aptitud')
    plt.plot(hist_prom,  color='#ec4899', linewidth=1.5, linestyle='--', label='Promedio')
    plt.plot(hist_peor,  color='#f97316', linewidth=1.5, linestyle=':',  label='Peor aptitud')
    plt.title("EVENTOESPACIO v2 — Evolución del Fitness")
    plt.xlabel("Generación"); plt.ylabel("Aptitud (0-1)"); plt.ylim(0, 1)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("evolucion_fitness_v2.png", dpi=150)
    plt.show()