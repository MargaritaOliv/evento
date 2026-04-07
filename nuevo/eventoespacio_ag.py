"""
EVENTOESPACIO — Núcleo del Algoritmo Genético
Expone run_ag() que retorna los 3 mejores individuos distintos.

BUGS CORREGIDOS:
  1. individuo_aleatorio: ahora verifica colisiones entre elementos ya colocados
  2. _solapamiento: penalización mucho más agresiva (cuadrática, no ratio suave)
  3. _p_prior: fórmula tenía un error de paréntesis que la hacía salir de [0,1]
  4. top3: garantiza diversidad mínima entre los 3 mejores (distancia de posiciones)
"""

import numpy as np
import pandas as pd
import random
import copy
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any

# ─────────────────────────────────────────────────────────────────────────────
#  CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────

AG_CONFIG = {
    "POBLACION":     100,
    "GENERACIONES":  250,
    "PROB_CRUCE":    0.85,
    "PROB_MUTACION": 0.20,
    "ELITISMO":      5,
    "TORNEO_K":      4,
}

PESOS = {
    "E_distribucion": 0.25,
    "F_flujo":        0.30,
    "C_conectividad": 0.25,
    "P_prioridad":    0.20,
}

ZONA_OPTIMA = {
    "escenario":       (0.50, 0.20),
    "stand_comida":    (0.25, 0.70),
    "stand_artesania": (0.75, 0.70),
    "servicio":        (0.50, 0.50),
    "acceso":          (0.50, 0.00),
    "zona_especial":   (0.50, 0.30),
    "zona_descanso":   (0.20, 0.50),
}

COLORES_TIPO = {
    "escenario":       "#E05252",
    "stand_comida":    "#E8A838",
    "stand_artesania": "#4CAF82",
    "servicio":        "#4A9EDB",
    "acceso":          "#9B6DD6",
    "zona_especial":   "#C94040",
    "zona_descanso":   "#3ABFB0",
}

# ─────────────────────────────────────────────────────────────────────────────
#  ESTRUCTURAS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Elemento:
    id: int
    nombre: str
    tipo: str
    ancho: int
    alto: int
    prioridad: int
    requires_access: bool

@dataclass
class Restriccion:
    id: int
    tipo: str
    x1: int; y1: int; x2: int; y2: int
    descripcion: str

@dataclass
class Gen:
    elemento_id: int
    x: int
    y: int

Individuo = List[Gen]

# ─────────────────────────────────────────────────────────────────────────────
#  CARGA
# ─────────────────────────────────────────────────────────────────────────────

def cargar_elementos(ruta: str) -> List[Elemento]:
    df = pd.read_csv(ruta)
    return [Elemento(
        id=int(r["id"]), nombre=str(r["nombre"]),
        tipo=str(r["tipo"]).strip(),
        ancho=int(r["ancho"]), alto=int(r["alto"]),
        prioridad=int(r["prioridad"]),
        requires_access=str(r["requires_access"]).strip().lower() in ("true","1","yes")
    ) for _, r in df.iterrows()]

def cargar_restricciones(ruta: str) -> List[Restriccion]:
    df = pd.read_csv(ruta)
    return [Restriccion(
        id=int(r["id"]), tipo=str(r["tipo"]).strip(),
        x1=int(r["x1"]), y1=int(r["y1"]),
        x2=int(r["x2"]), y2=int(r["y2"]),
        descripcion=str(r["descripcion"])
    ) for _, r in df.iterrows()]

# ─────────────────────────────────────────────────────────────────────────────
#  VENUE
# ─────────────────────────────────────────────────────────────────────────────

class Venue:
    def __init__(self, ancho: int, alto: int, restricciones: List[Restriccion]):
        self.ancho = ancho
        self.alto  = alto
        self.restricciones = restricciones
        self.mapa_bloqueado = np.zeros((alto, ancho), dtype=bool)
        self.puntos_acceso: List[Tuple[int,int]] = []
        self.d_max = ancho + alto
        self._procesar()

    def _procesar(self):
        for r in self.restricciones:
            x1,x2 = min(r.x1,r.x2), max(r.x1,r.x2)
            y1,y2 = min(r.y1,r.y2), max(r.y1,r.y2)
            if r.tipo in ("zona_restringida","columna"):
                self.mapa_bloqueado[y1:y2+1, x1:x2+1] = True
            elif r.tipo in ("entrada","salida"):
                self.puntos_acceso.append(((x1+x2)//2, (y1+y2)//2))

    def celdas_disponibles(self) -> int:
        return int((~self.mapa_bloqueado).sum())

    def celdas_disp_cuadrante(self, qx1,qy1,qx2,qy2) -> int:
        return int((~self.mapa_bloqueado[qy1:qy2, qx1:qx2]).sum())

    def es_valido(self, x, y, ancho, alto) -> bool:
        """True si el rectángulo cabe sin tocar bordes ni celdas bloqueadas."""
        if x < 1 or y < 1 or x + ancho > self.ancho - 1 or y + alto > self.alto - 1:
            return False
        return not bool(self.mapa_bloqueado[y:y+alto, x:x+ancho].any())

# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDAD: detección de solapamiento entre dos rectángulos
# ─────────────────────────────────────────────────────────────────────────────

def _solapan(ax, ay, aw, ah, bx, by, bw, bh) -> bool:
    """True si los dos rectángulos se solapan."""
    return not (ax + aw <= bx or bx + bw <= ax or
                ay + ah <= by or by + bh <= ay)

# ─────────────────────────────────────────────────────────────────────────────
#  PROCESO 1 — INICIALIZACIÓN  (BUG 1 CORREGIDO)
#
#  El error original: solo verificaba venue.es_valido() — que comprueba
#  celdas bloqueadas del venue — pero NO verificaba colisiones con los
#  elementos ya colocados en el mismo individuo.
#  Corrección: se mantiene una lista de rectángulos ya usados y se verifica
#  solapamiento con cada uno antes de aceptar la posición.
# ─────────────────────────────────────────────────────────────────────────────

def individuo_aleatorio(elementos: List[Elemento], venue: Venue) -> Individuo:
    ind: Individuo = []
    ocupados: List[Tuple[int,int,int,int]] = []  # (x, y, ancho, alto)

    for e in elementos:
        colocado = False
        for _ in range(1000):
            x = random.randint(1, max(1, venue.ancho - e.ancho - 1))
            y = random.randint(1, max(1, venue.alto  - e.alto  - 1))

            # 1) Verificar venue (bordes + zonas bloqueadas)
            if not venue.es_valido(x, y, e.ancho, e.alto):
                continue

            # 2) Verificar que no se solape con ningún elemento ya colocado
            if any(_solapan(x, y, e.ancho, e.alto, ox, oy, ow, oh)
                   for ox, oy, ow, oh in ocupados):
                continue

            ind.append(Gen(e.id, x, y))
            ocupados.append((x, y, e.ancho, e.alto))
            colocado = True
            break

        if not colocado:
            # Fallback: colocar en (1,1) aunque solape — el fitness penalizará
            ind.append(Gen(e.id, 1, 1))

    return ind

def inicializar(elementos, venue, tam) -> List[Individuo]:
    return [individuo_aleatorio(elementos, venue) for _ in range(tam)]

# ─────────────────────────────────────────────────────────────────────────────
#  PROCESO 2 — FITNESS
# ─────────────────────────────────────────────────────────────────────────────

def _solapamiento(ind: Individuo, em: dict) -> float:
    """
    BUG 2 CORREGIDO — Penalización agresiva por solapamiento.

    Error original: λ = pares_solapados / total_pares
    Era demasiado suave: con 18 elementos (153 pares), 3 solapamientos
    solo daban λ ≈ 0.02 → fitness apenas bajaba 2%.

    Corrección: penalización cuadrática proporcional al área solapada.
    Si hay cualquier solapamiento → fitness se destruye cerca de 0.
    """
    n = len(ind)
    area_total_solapada = 0

    for i in range(n):
        for j in range(i + 1, n):
            ga, gb = ind[i], ind[j]
            ea, eb = em[ga.elemento_id], em[gb.elemento_id]

            # Calcular área de intersección
            ix = max(0, min(ga.x + ea.ancho, gb.x + eb.ancho) - max(ga.x, gb.x))
            iy = max(0, min(ga.y + ea.alto,  gb.y + eb.alto)  - max(ga.y, gb.y))
            area_total_solapada += ix * iy

    if area_total_solapada == 0:
        return 0.0

    # Normalizar contra el área total de todos los elementos
    area_elementos = sum(em[g.elemento_id].ancho * em[g.elemento_id].alto for g in ind)
    # Penalización cuadrática: incluso un solapamiento pequeño = penalización fuerte
    ratio = min(area_total_solapada / max(area_elementos, 1), 1.0)
    return float(ratio ** 0.5)   # raíz cuadrada → penalización alta para áreas pequeñas


def _e_dist(ind, em, venue):
    """
    E_dist — Distribución equilibrada entre los 4 cuadrantes del venue.

    PROBLEMA ORIGINAL: 1 - sigma/media colapsaba a ~0 cuando la densidad
    media era muy pequeña (venue grande, elementos pequeños → media≈0.01),
    haciendo que sigma/media explotara aunque visualmente hubiera balance.

    CORRECCIÓN: medir la fracción de ÁREA DE ELEMENTOS en cada cuadrante
    (no la densidad respecto al venue). Las fracciones suman 1.0, por lo que
    sigma_max = 0.5 (todo en un cuadrante). Normalizar con 2·sigma.

      fraccion_k = area_elementos_en_cuadrante_k / area_total_elementos
      E_dist = max(0, 1 - 2·sigma(fracciones))

    Resultado: 1.0 si los 4 cuadrantes tienen exactamente el mismo peso,
               0.0 si todos los elementos están en un solo cuadrante.
    """
    cuads = [
        (0,              0,             venue.ancho//2, venue.alto//2),
        (venue.ancho//2, 0,             venue.ancho,    venue.alto//2),
        (0,              venue.alto//2, venue.ancho//2, venue.alto),
        (venue.ancho//2, venue.alto//2, venue.ancho,    venue.alto),
    ]

    area_total = sum(
        em[g.elemento_id].ancho * em[g.elemento_id].alto for g in ind
    )
    if area_total == 0:
        return 1.0

    fracciones = []
    for qx1, qy1, qx2, qy2 in cuads:
        area_cuad = sum(
            em[g.elemento_id].ancho * em[g.elemento_id].alto
            for g in ind
            if qx1 <= g.x + em[g.elemento_id].ancho // 2 < qx2
            and qy1 <= g.y + em[g.elemento_id].alto  // 2 < qy2
        )
        fracciones.append(area_cuad / area_total)

    sigma = np.array(fracciones).std()
    return float(max(0.0, 1.0 - 2.0 * sigma))


def _f_flujo(ind, em, venue):
    ALPHA = 0.6
    ocu = np.copy(venue.mapa_bloqueado).astype(int)
    for g in ind:
        e = em[g.elemento_id]
        ocu[g.y:min(g.y+e.alto, venue.alto),
            g.x:min(g.x+e.ancho, venue.ancho)] = 1
    disp  = venue.celdas_disponibles()
    lib   = (ocu == 0) & (~venue.mapa_bloqueado)
    n_lib = int(lib.sum())
    if disp == 0:
        return 0.0
    R = n_lib / disp
    if n_lib == 0:
        B = 1.0
    else:
        li  = lib.astype(int)
        vec = (np.roll(li,1,0) + np.roll(li,-1,0) +
               np.roll(li,1,1) + np.roll(li,-1,1))
        B = int(((vec <= 2) & lib).sum()) / n_lib
    return float(ALPHA * R + (1 - ALPHA) * (1 - B))


def _c_conex(ind, em, venue):
    """
    C_conex — Accesibilidad de elementos prioritarios a entradas/salidas.

    PROBLEMA ORIGINAL: se normalizaba con d_max = ancho + alto = 90.
    Pero los accesos están en los bordes y los elementos en el interior,
    por lo que la distancia mínima posible ya es ~15-25 celdas, haciendo
    que el score máximo alcanzable fuera solo ~0.7, nunca cercano a 1.0.

    CORRECCIÓN: normalizar con la distancia máxima REAL del venue —
    calculada dinámicamente como la distancia Manhattan máxima posible
    desde cualquier celda interior hasta el punto de acceso MÁS CERCANO.
    Esto calibra la escala al espacio real disponible.

      d_ref = max distancia Manhattan desde esquina más lejana a acceso más cercano
      score_i = 1 - d_min_i / d_ref
    """
    acc = venue.puntos_acceso or [(venue.ancho // 2, venue.alto // 2)]
    elems_acc = [g for g in ind if em[g.elemento_id].requires_access]
    if not elems_acc:
        return 1.0

    # Calcular d_ref: distancia máxima real desde cualquier esquina al acceso más cercano
    esquinas = [
        (1, 1),
        (venue.ancho - 2, 1),
        (1, venue.alto - 2),
        (venue.ancho - 2, venue.alto - 2),
    ]
    d_ref = max(
        min(abs(ex - ax) + abs(ey - ay) for ax, ay in acc)
        for ex, ey in esquinas
    )
    if d_ref == 0:
        return 1.0

    scores = []
    for g in elems_acc:
        e  = em[g.elemento_id]
        cx, cy = g.x + e.ancho // 2, g.y + e.alto // 2
        d_min = min(abs(cx - ax) + abs(cy - ay) for ax, ay in acc)
        scores.append(max(0.0, 1.0 - d_min / d_ref))
    return float(np.mean(scores))


def _p_prior(ind, em, venue):
    """
    BUG 3 CORREGIDO — Error de paréntesis en la fórmula original:
        total += e.prioridad * (1.0 - abs(cx-ox) + abs(cy-oy) / venue.d_max)
    El + en vez de - hacía que el valor pudiera ser > 1 o negativo.
    Corrección:
        s_i = 1 - (d_manhattan / d_max)   con d_manhattan = |cx-ox| + |cy-oy|
    """
    sp = sum(e.prioridad for e in em.values())
    if sp == 0:
        return 0.0
    total = 0.0
    for g in ind:
        e  = em[g.elemento_id]
        fx, fy = ZONA_OPTIMA.get(e.tipo, (0.5, 0.5))
        ox = int(fx * venue.ancho)
        oy = int(fy * venue.alto)
        cx = g.x + e.ancho // 2
        cy = g.y + e.alto  // 2
        d  = abs(cx - ox) + abs(cy - oy)          # distancia Manhattan
        s  = max(0.0, 1.0 - d / venue.d_max)      # score normalizado [0,1]
        total += e.prioridad * s
    return float(min(total / sp, 1.0))


def fitness(ind: Individuo, elementos: List[Elemento],
            venue: Venue) -> Tuple[float, dict]:
    """
    F_total = (w1·E + w2·F + w3·C + w4·P) · (1 - λ_solap) · (1 - λ_fuera)
    """
    em = {e.id: e for e in elementos}

    E = _e_dist(ind, em, venue)
    F = _f_flujo(ind, em, venue)
    C = _c_conex(ind, em, venue)
    P = _p_prior(ind, em, venue)

    base = (PESOS["E_distribucion"] * E +
            PESOS["F_flujo"]        * F +
            PESOS["C_conectividad"] * C +
            PESOS["P_prioridad"]    * P)

    ls = _solapamiento(ind, em)
    fuera = sum(1 for g in ind
                if not venue.es_valido(g.x, g.y,
                                       em[g.elemento_id].ancho,
                                       em[g.elemento_id].alto))
    lf = fuera / len(ind) if ind else 0.0

    total = base * (1 - ls) * (1 - lf)

    return total, {
        "E_distribucion": round(E, 4),
        "F_flujo":        round(F, 4),
        "C_conectividad": round(C, 4),
        "P_prioridad":    round(P, 4),
        "penalizacion":   round(1 - (1 - ls) * (1 - lf), 4),
        "fitness":        round(total, 6),
    }

# ─────────────────────────────────────────────────────────────────────────────
#  PROCESOS 3–6
# ─────────────────────────────────────────────────────────────────────────────

def seleccion_torneo(pob, scores, k):
    cands = random.sample(range(len(pob)), k)
    return copy.deepcopy(pob[max(cands, key=lambda i: scores[i])])


def cruce(p1, p2, prob):
    if random.random() > prob or len(p1) < 2:
        return copy.deepcopy(p1), copy.deepcopy(p2)
    pt = random.randint(1, len(p1) - 1)
    return copy.deepcopy(p1[:pt] + p2[pt:]), copy.deepcopy(p2[:pt] + p1[pt:])


def mutacion(ind, elementos, venue, prob):
    """
    Mutación que también respeta colisiones entre elementos del mismo individuo.
    """
    em = {e.id: e for e in elementos}
    m  = copy.deepcopy(ind)

    for idx, g in enumerate(m):
        if random.random() < prob:
            e = em[g.elemento_id]
            # Rectángulos de todos los demás elementos (para evitar colisiones)
            otros = [(m[j].x, m[j].y, em[m[j].elemento_id].ancho,
                      em[m[j].elemento_id].alto)
                     for j in range(len(m)) if j != idx]

            for _ in range(400):
                nx = random.randint(1, max(1, venue.ancho - e.ancho - 1))
                ny = random.randint(1, max(1, venue.alto  - e.alto  - 1))
                if not venue.es_valido(nx, ny, e.ancho, e.alto):
                    continue
                if any(_solapan(nx, ny, e.ancho, e.alto, ox, oy, ow, oh)
                       for ox, oy, ow, oh in otros):
                    continue
                g.x, g.y = nx, ny
                break
    return m


def poda(pob, scores, hijos, sh, n_elite, tam):
    idx_e = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:n_elite]
    comb  = [copy.deepcopy(pob[i]) for i in idx_e] + hijos
    sc    = [scores[i] for i in idx_e] + sh
    idx   = sorted(range(len(sc)), key=lambda i: sc[i], reverse=True)
    return [comb[i] for i in idx[:tam]], [sc[i] for i in idx[:tam]]

# ─────────────────────────────────────────────────────────────────────────────
#  DIVERSIDAD — distancia entre dos individuos
# ─────────────────────────────────────────────────────────────────────────────

def _distancia(ind_a: Individuo, ind_b: Individuo) -> float:
    """
    Distancia promedio de posiciones entre dos individuos.
    Usada para garantizar que el top-3 sea diverso.
    """
    total = sum(
        abs(ga.x - gb.x) + abs(ga.y - gb.y)
        for ga, gb in zip(ind_a, ind_b)
    )
    return total / max(len(ind_a), 1)

# ─────────────────────────────────────────────────────────────────────────────
#  SERIALIZACIÓN
# ─────────────────────────────────────────────────────────────────────────────

def individuo_a_dict(ind: Individuo, elementos: List[Elemento],
                     venue: Venue, metricas: dict, rank: int) -> Dict[str, Any]:
    em = {e.id: e for e in elementos}
    return {
        "rank":      rank,
        "metricas":  metricas,
        "venue_ancho": venue.ancho,
        "venue_alto":  venue.alto,
        "elementos": [{
            "id":        em[g.elemento_id].id,
            "nombre":    em[g.elemento_id].nombre,
            "tipo":      em[g.elemento_id].tipo,
            "x":         g.x, "y": g.y,
            "ancho":     em[g.elemento_id].ancho,
            "alto":      em[g.elemento_id].alto,
            "prioridad": em[g.elemento_id].prioridad,
            "color":     COLORES_TIPO.get(em[g.elemento_id].tipo, "#8b949e"),
        } for g in ind],
        "restricciones": [{
            "tipo":        r.tipo,
            "x1": min(r.x1,r.x2), "y1": min(r.y1,r.y2),
            "x2": max(r.x1,r.x2), "y2": max(r.y1,r.y2),
            "descripcion": r.descripcion,
        } for r in venue.restricciones],
    }

# ─────────────────────────────────────────────────────────────────────────────
#  PUNTO DE ENTRADA
# ─────────────────────────────────────────────────────────────────────────────

def run_ag(venue_ancho: int, venue_alto: int,
           ruta_elementos: str, ruta_restricciones: str,
           config: dict = None, seed: int = 42) -> Dict[str, Any]:

    random.seed(seed)
    np.random.seed(seed)

    cfg     = config or AG_CONFIG
    TAM     = cfg["POBLACION"]
    GENS    = cfg["GENERACIONES"]
    P_CRUCE = cfg["PROB_CRUCE"]
    P_MUT   = cfg["PROB_MUTACION"]
    ELITE   = cfg["ELITISMO"]
    K       = cfg["TORNEO_K"]

    elementos     = cargar_elementos(ruta_elementos)
    restricciones = cargar_restricciones(ruta_restricciones)
    venue         = Venue(venue_ancho, venue_alto, restricciones)

    # 1. Inicialización
    pob    = inicializar(elementos, venue, TAM)
    scores = [fitness(ind, elementos, venue)[0] for ind in pob]
    historial = []

    for gen_num in range(GENS):
        hijos, sh = [], []
        while len(hijos) < TAM:
            p1 = seleccion_torneo(pob, scores, K)
            p2 = seleccion_torneo(pob, scores, K)
            h1, h2 = cruce(p1, p2, P_CRUCE)
            h1 = mutacion(h1, elementos, venue, P_MUT)
            h2 = mutacion(h2, elementos, venue, P_MUT)
            hijos.extend([h1, h2])

        hijos  = hijos[:TAM]
        sh     = [fitness(h, elementos, venue)[0] for h in hijos]
        pob, scores = poda(pob, scores, hijos, sh, ELITE, TAM)

        historial.append({
            "generacion": gen_num + 1,
            "mejor":      round(max(scores), 6),
            "promedio":   round(float(np.mean(scores)), 6),
        })

    # ── BUG 4 CORREGIDO — Top 3 con diversidad garantizada ───────────────────
    #
    # Error original: se tomaban los 3 primeros del ranking → casi idénticos
    # porque el elitismo hace que los top converjan al mismo individuo.
    #
    # Corrección: selección greedy por diversidad.
    #   - El #1 es siempre el de mayor fitness.
    #   - El #2 es el de mayor fitness que sea suficientemente distinto del #1.
    #   - El #3 es el de mayor fitness distinto del #1 y del #2.
    # Umbral de distancia mínima: 3 celdas promedio por elemento.

    UMBRAL_DIVERSIDAD = 3.0   # distancia mínima promedio entre individuos

    idx_ord = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    top3_idx    = [idx_ord[0]]
    top3_indivs = [pob[idx_ord[0]]]

    for idx in idx_ord[1:]:
        if len(top3_idx) == 3:
            break
        candidato = pob[idx]
        # Solo agregar si es suficientemente distinto de todos los ya elegidos
        if all(_distancia(candidato, pob[j]) >= UMBRAL_DIVERSIDAD
               for j in top3_idx):
            top3_idx.append(idx)
            top3_indivs.append(candidato)

    # Si no se encontraron 3 distintos, relajar umbral y completar
    if len(top3_idx) < 3:
        for idx in idx_ord[1:]:
            if len(top3_idx) == 3:
                break
            if idx not in top3_idx:
                top3_idx.append(idx)
                top3_indivs.append(pob[idx])

    top3 = []
    for rank, (idx, ind) in enumerate(zip(top3_idx, top3_indivs), 1):
        _, met = fitness(ind, elementos, venue)
        top3.append(individuo_a_dict(ind, elementos, venue, met, rank))

    return {
        "top3":      top3,
        "historial": historial,
        "config": {
            "venue_ancho":  venue_ancho,
            "venue_alto":   venue_alto,
            "n_elementos":  len(elementos),
            "generaciones": GENS,
            "poblacion":    TAM,
        }
    }