
import csv
import random
import math
import copy
import os

# ─────────────────────────────────────────────
# RUTAS DE ARCHIVOS
# ─────────────────────────────────────────────
BASE_DIR          = os.path.dirname(os.path.abspath(__file__))
DATA_DIR          = os.path.join(BASE_DIR, "data")
ARCHIVO_ELEMENTOS     = os.path.join(DATA_DIR, "catalogo_elementos.csv")
ARCHIVO_RESTRICCIONES = os.path.join(DATA_DIR, "catalogo_restricciones.csv")

# ─────────────────────────────────────────────
# 1. PARÁMETROS DEL ALGORITMO GENÉTICO
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
    """Lee el catálogo de elementos desde un CSV."""
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
    """Lee el catálogo de restricciones del espacio desde un CSV."""
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
    """Filtra solo las entradas del recinto."""
    return [r for r in restricciones if r["tipo"] == "entrada"]


def obtener_celdas_restringidas(restricciones):
    """
    Devuelve el conjunto de coordenadas bloqueadas.
    Cada zona_restringida ocupa 1x1 celda.
    """
    return {(r["x"], r["y"]) for r in restricciones if r["tipo"] == "zona_restringida"}


# ─────────────────────────────────────────────
# 3. UTILIDADES GEOMÉTRICAS (CORRECCIÓN #2 y #3)
# ─────────────────────────────────────────────
def celdas_del_elemento(x, y, ancho, alto):
    """
    Retorna el conjunto de todas las celdas que ocupa un elemento.
    FIX #2: valida el cuerpo completo, no solo la esquina (x,y).
    """
    return {(x + dx, y + dy)
            for dx in range(ancho)
            for dy in range(alto)}


def elemento_es_valido(x, y, ancho, alto, celdas_restringidas):
    """
    Verifica que ninguna celda del elemento caiga en zona restringida.
    FIX #2: antes solo se verificaba el punto (x, y).
    """
    return celdas_del_elemento(x, y, ancho, alto).isdisjoint(celdas_restringidas)


def se_solapan(x1, y1, w1, h1, x2, y2, w2, h2):
    """Retorna True si dos rectángulos se solapan."""
    return not (x1 + w1 <= x2 or x2 + w2 <= x1 or
                y1 + h1 <= y2 or y2 + h2 <= y1)


def hay_solapamiento(individuo, elementos):
    """
    FIX #3: detecta si algún par de elementos se superpone.
    Se usa para penalizar en la función de aptitud.
    """
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
    """
    Crea un individuo aleatorio.
    FIX #2: valida el cuerpo completo del elemento, no solo (x,y).
    """
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    individuo = []
    for elem in elementos:
        intentos = 0
        while True:
            intentos += 1
            x = random.randint(0, W - elem["ancho"])
            y = random.randint(0, H - elem["alto"])
            if elemento_es_valido(x, y, elem["ancho"], elem["alto"], celdas_restringidas):
                break
            # Seguridad: si hay demasiados intentos fallidos, coloca sin validar
            if intentos > 500:
                break
        individuo.append((x, y))
    return individuo


def crear_poblacion(elementos, celdas_restringidas, tam=None, W=None, H=None):
    """Crea la población inicial."""
    if tam is None: tam = TAM_POBLACION
    return [crear_individuo(elementos, celdas_restringidas, W, H)
            for _ in range(tam)]


# ─────────────────────────────────────────────
# 5. FUNCIONES OBJETIVO
# ─────────────────────────────────────────────
def calcular_O1_distribucion(individuo, W=None, H=None):
    """O1 — Distribución equilibrada del espacio en 4 cuadrantes."""
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
    """O2 — Calidad del flujo de personas (espacio libre)."""
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    celdas_totales  = W * H
    celdas_ocupadas = sum(e["ancho"] * e["alto"] for e in elementos)
    O2 = (celdas_totales - celdas_ocupadas) / celdas_totales
    return max(0.0, min(1.0, O2))


def calcular_O3_conectividad(individuo, elementos, entradas, W=None, H=None):
    """O3 — Accesibilidad: elementos con acceso cerca de entradas."""
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
    """O4 — Elementos de alta prioridad en zona central."""
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
    """
    Función de aptitud: O1 × O2 × O3 × O4.
    FIX #3: retorna 0 si hay solapamiento entre elementos.
    """
    if W is None: W = ANCHO_GRID
    if H is None: H = ALTO_GRID

    # Penalización total si hay solapamiento
    if hay_solapamiento(individuo, elementos):
        return 0.0

    O1 = calcular_O1_distribucion(individuo, W, H)
    O2 = calcular_O2_flujo(individuo, elementos, W, H)
    O3 = calcular_O3_conectividad(individuo, elementos, entradas, W, H)
    O4 = calcular_O4_prioridad(individuo, elementos, W, H)

    return O1 * O2 * O3 * O4


# ─────────────────────────────────────────────
# 6. SELECCIÓN POR TORNEO (FIX #4: k=3)
# ─────────────────────────────────────────────
def seleccion_torneo(poblacion, aptitudes, k=3):
    """
    Selecciona un individuo por torneo de tamaño k=3.
    FIX #4: mayor presión selectiva que el torneo de tamaño 2 original.
    """
    candidatos = random.sample(range(len(poblacion)), min(k, len(poblacion)))
    mejor_idx  = max(candidatos, key=lambda i: aptitudes[i])
    return copy.deepcopy(poblacion[mejor_idx])


# ─────────────────────────────────────────────
# 7. CRUZAMIENTO
# ─────────────────────────────────────────────
def cruzamiento(padre1, padre2, p_cruza=None):
    """Cruzamiento de un punto."""
    if p_cruza is None: p_cruza = P_CRUZA
    if random.random() < p_cruza:
        punto = random.randint(1, len(padre1) - 1)
        return padre1[:punto] + padre2[punto:]
    return copy.deepcopy(padre1)


# ─────────────────────────────────────────────
# 8. MUTACIÓN (FIX #2: valida cuerpo completo)
# ─────────────────────────────────────────────
def mutacion(individuo, elementos, celdas_restringidas, pmi=None, pmg=None, W=None, H=None):
    """
    Mutación por gen.
    FIX #2: valida que el cuerpo completo del elemento no toque celdas restringidas.
    """
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
                    x = random.randint(0, W - elementos[i]["ancho"])
                    y = random.randint(0, H - elementos[i]["alto"])
                    if elemento_es_valido(x, y, elementos[i]["ancho"],
                                         elementos[i]["alto"], celdas_restringidas):
                        break
                    if intentos > 500:
                        break
                individuo[i] = (x, y)
    return individuo


# ─────────────────────────────────────────────
# 9. ALGORITMO GENÉTICO PRINCIPAL (FIX #5: elitismo estricto)
# ─────────────────────────────────────────────
def algoritmo_genetico(elementos, entradas, celdas_restringidas, params=None):
    """
    Ejecuta el AG completo.
    FIX #5: elitismo estricto garantiza que la solución nunca empeora.
    Acepta parámetros externos desde la interfaz web.
    """
    # Parámetros: usa los de la interfaz o los globales por defecto
    tam_pob = params.get("tam_poblacion", TAM_POBLACION) if params else TAM_POBLACION
    pc      = params.get("p_cruza",       P_CRUZA)       if params else P_CRUZA
    pmi     = params.get("p_mut_ind",     P_MUT_IND)     if params else P_MUT_IND
    pmg     = params.get("p_mut_gen",     P_MUT_GEN)     if params else P_MUT_GEN
    n_gen   = params.get("generaciones",  N_GENERACIONES) if params else N_GENERACIONES
    W       = params.get("ancho",         ANCHO_GRID)    if params else ANCHO_GRID
    H       = params.get("alto",          ALTO_GRID)     if params else ALTO_GRID

    # Inicialización
    poblacion = crear_poblacion(elementos, celdas_restringidas, tam_pob, W, H)

    historial_mejor    = []
    historial_promedio = []
    mejor_individuo    = None
    mejor_aptitud      = -1.0

    for generacion in range(n_gen):

        # Evaluación
        aptitudes   = [calcular_aptitud(ind, elementos, entradas, W, H) for ind in poblacion]
        mejor_gen   = max(aptitudes)
        promedio_gen = sum(aptitudes) / len(aptitudes)

        historial_mejor.append(round(mejor_gen, 6))
        historial_promedio.append(round(promedio_gen, 6))

        # FIX #5: actualizar mejor global solo si mejora
        idx_mejor = aptitudes.index(mejor_gen)
        if mejor_gen > mejor_aptitud:
            mejor_aptitud   = mejor_gen
            mejor_individuo = copy.deepcopy(poblacion[idx_mejor])

        # Nueva generación con elitismo estricto
        nueva_poblacion = [copy.deepcopy(mejor_individuo)]  # siempre el mejor histórico

        while len(nueva_poblacion) < tam_pob:
            padre1 = seleccion_torneo(poblacion, aptitudes)
            padre2 = seleccion_torneo(poblacion, aptitudes)
            hijo   = cruzamiento(padre1, padre2, pc)
            hijo   = mutacion(hijo, elementos, celdas_restringidas, pmi, pmg, W, H)
            nueva_poblacion.append(hijo)

        poblacion = nueva_poblacion

    return mejor_individuo, mejor_aptitud, historial_mejor, historial_promedio


# ─────────────────────────────────────────────
# 10. FUNCIÓN PÚBLICA PARA app.py
# ─────────────────────────────────────────────
def ejecutar_ag(params):
    """
    Punto de entrada único que usa app.py.
    Carga datos, ejecuta el AG y devuelve el resultado listo para JSON.
    """
    W = params.get("ancho", ANCHO_GRID)
    H = params.get("alto",  ALTO_GRID)

    elementos     = cargar_elementos()
    restricciones = cargar_restricciones()
    entradas      = obtener_entradas(restricciones)
    celdas_rest   = obtener_celdas_restringidas(restricciones)

    mejor_ind, mejor_apt, hist_mejor, hist_promedio = \
        algoritmo_genetico(elementos, entradas, celdas_rest, params)

    O1 = round(calcular_O1_distribucion(mejor_ind, W, H), 4)
    O2 = round(calcular_O2_flujo(mejor_ind, elementos, W, H), 4)
    O3 = round(calcular_O3_conectividad(mejor_ind, elementos, entradas, W, H), 4)
    O4 = round(calcular_O4_prioridad(mejor_ind, elementos, W, H), 4)

    tabla = []
    for i, e in enumerate(elementos):
        x, y = mejor_ind[i]
        tabla.append({
            "id":        e["id"],
            "tipo":      e["tipo"],
            "prioridad": e["prioridad"],
            "x": x, "y": y,
            "ancho": e["ancho"],
            "alto":  e["alto"]
        })

    return {
        "aptitud":        round(mejor_apt, 6),
        "O1": O1, "O2": O2, "O3": O3, "O4": O4,
        "hist_mejor":     hist_mejor,
        "hist_promedio":  hist_promedio,
        "tabla":          tabla,
        "restricciones":  restricciones,
        "ancho": W, "alto": H
    }


# ─────────────────────────────────────────────
# 11. PUNTO DE ENTRADA (ejecución directa)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    elementos     = cargar_elementos()
    restricciones = cargar_restricciones()
    entradas      = obtener_entradas(restricciones)
    celdas_rest   = obtener_celdas_restringidas(restricciones)

    print(f"Elementos cargados   : {len(elementos)}")
    print(f"Restricciones        : {len(restricciones)}")
    print(f"Entradas del recinto : {len(entradas)}")

    mejor_ind, mejor_apt, hist_mejor, hist_prom = \
        algoritmo_genetico(elementos, entradas, celdas_rest)

    # Tabla resumen
    print("\n" + "=" * 60)
    print("  TABLA DE RESULTADOS — MEJOR DISTRIBUCIÓN ENCONTRADA")
    print("=" * 60)
    print(f"{'ID':<5} {'Tipo':<20} {'Prioridad':<12} {'X':<5} {'Y':<5}")
    print("-" * 60)
    for i, elem in enumerate(elementos):
        x, y = mejor_ind[i]
        print(f"{elem['id']:<5} {elem['tipo']:<20} {elem['prioridad']:<12} {x:<5} {y:<5}")
    print("-" * 60)
    print(f"  ★ APTITUD FINAL = {mejor_apt:.6f}")
    print("=" * 60)

    # Gráfica de evolución
    plt.figure(figsize=(10, 5))
    plt.plot(hist_mejor, color='#a855f7', linewidth=2, label='Mejor aptitud')
    plt.plot(hist_prom,  color='#ec4899', linewidth=1.5, linestyle='--', label='Aptitud promedio')
    plt.title("EVENTOESPACIO — Evolución del Fitness", fontsize=14, fontweight='bold')
    plt.xlabel("Generación"); plt.ylabel("Aptitud (0 - 1)"); plt.ylim(0, 1)
    plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig("evolucion_fitness.png", dpi=150); plt.show()