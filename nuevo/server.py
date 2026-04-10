from flask import Flask, render_template, request, jsonify
import os
import traceback
from eventoespacio_ag import (
    ejecutar_ag,
    TAM_POBLACION, P_CRUZA, P_MUT_IND, P_MUT_GEN, N_GENERACIONES,
    ANCHO_GRID, ALTO_GRID,
    W_DISTRIBUCION, W_FLUJO, W_CONECTIVIDAD, W_PRIORIDAD
)

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configuración del AG expuesta al frontend (construida desde las constantes reales)
AG_CONFIG = {
    "TAM_POBLACION":  TAM_POBLACION,
    "P_CRUZA":        P_CRUZA,
    "P_MUT_IND":      P_MUT_IND,
    "P_MUT_GEN":      P_MUT_GEN,
    "N_GENERACIONES": N_GENERACIONES,
    "ANCHO_GRID":     ANCHO_GRID,
    "ALTO_GRID":      ALTO_GRID,
    "W_DISTRIBUCION": W_DISTRIBUCION,
    "W_FLUJO":        W_FLUJO,
    "W_CONECTIVIDAD": W_CONECTIVIDAD,
    "W_PRIORIDAD":    W_PRIORIDAD,
}

# Datasets disponibles con sus dimensiones y rutas de archivos
DATASETS = {
    "chico": {
        "ancho": 50,
        "alto":  40,
        "archivo_elementos":     os.path.join(BASE_DIR, "data", "elementos",     "chico.csv"),
        "archivo_restricciones": os.path.join(BASE_DIR, "data", "restricciones", "chico.csv"),
    },
    "mediano": {
        "ancho": 80,
        "alto":  60,
        "archivo_elementos":     os.path.join(BASE_DIR, "data", "elementos",     "mediano.csv"),
        "archivo_restricciones": os.path.join(BASE_DIR, "data", "restricciones", "mediano.csv"),
    },
    "grande": {
        "ancho": 120,
        "alto":  90,
        "archivo_elementos":     os.path.join(BASE_DIR, "data", "elementos",     "grande.csv"),
        "archivo_restricciones": os.path.join(BASE_DIR, "data", "restricciones", "grande.csv"),
    },
}


@app.route("/")
def index():
    # AG_CONFIG se pasa a Jinja2 para que el HTML pueda mostrar
    # los valores por defecto sin hardcodearlos en el frontend
    return render_template("index.html", config=AG_CONFIG)


@app.route("/ejecutar", methods=["POST"])
def ejecutar():
    data = request.get_json()

    tamano = data.get("tamano", "chico")

    if tamano not in DATASETS:
        return jsonify({"error": f"Tamaño inválido: '{tamano}'. Opciones: chico, mediano, grande"}), 400

    ds = DATASETS[tamano]

    # Verificar que los archivos existen antes de correr el AG
    for ruta in (ds["archivo_elementos"], ds["archivo_restricciones"]):
        if not os.path.exists(ruta):
            return jsonify({"error": f"Archivo no encontrado: {ruta}"}), 500

    # Construir params para ejecutar_ag
    # El frontend puede sobreescribir cualquier valor; si no lo manda, se usa el default del AG
    params = {
        "ancho":                 ds["ancho"],
        "alto":                  ds["alto"],
        "archivo_elementos":     ds["archivo_elementos"],
        "archivo_restricciones": ds["archivo_restricciones"],
        "tam_poblacion": int(data.get("tam_poblacion",  TAM_POBLACION)),
        "p_cruza":       float(data.get("p_cruza",      P_CRUZA)),
        "p_mut_ind":     float(data.get("p_mut_ind",    P_MUT_IND)),
        "p_mut_gen":     float(data.get("p_mut_gen",    P_MUT_GEN)),
        "generaciones":  int(data.get("generaciones",   N_GENERACIONES)),
        "wE":            float(data.get("wE",           W_DISTRIBUCION)),
        "wF":            float(data.get("wF",           W_FLUJO)),
        "wC":            float(data.get("wC",           W_CONECTIVIDAD)),
        "wP":            float(data.get("wP",           W_PRIORIDAD)),
    }

    try:
        resultado = ejecutar_ag(params)
        # Devuelve el resultado directamente (sin envoltorio ok/data)
        # para que sea compatible con lo que espera el index.html
        return jsonify(resultado)
    except Exception:
        error_detalle = traceback.format_exc()
        print(error_detalle)
        return jsonify({"error": error_detalle}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)