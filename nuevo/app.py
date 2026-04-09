"""
=============================================================
  EVENTOESPACIO — Back-end Flask  (v2)
  Ruta:  POST /ejecutar  (application/json)
  Campos: tamano ("chico"|"mediano"|"grande"),
          tam_poblacion, p_cruza, p_mut_ind, p_mut_gen,
          generaciones, wE, wF, wC, wP
  El backend resuelve automáticamente los CSV y dimensiones
  del venue según el tamano seleccionado.
=============================================================
"""

import os
import traceback
from flask import Flask, render_template, request, jsonify
from eventoespacio_ag import ejecutar_ag

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dimensiones y rutas de datos por tamaño de evento
DATASETS = {
    "chico": {
        "ancho": 50, "alto": 40,
        "archivo_elementos":     os.path.join(BASE_DIR, "data", "elementos",     "chico.csv"),
        "archivo_restricciones": os.path.join(BASE_DIR, "data", "restricciones", "chico.csv"),
    },
    "mediano": {
        "ancho": 80, "alto": 60,
        "archivo_elementos":     os.path.join(BASE_DIR, "data", "elementos",     "mediano.csv"),
        "archivo_restricciones": os.path.join(BASE_DIR, "data", "restricciones", "mediano.csv"),
    },
    "grande": {
        "ancho": 120, "alto": 90,
        "archivo_elementos":     os.path.join(BASE_DIR, "data", "elementos",     "grande.csv"),
        "archivo_restricciones": os.path.join(BASE_DIR, "data", "restricciones", "grande.csv"),
    },
}


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ejecutar", methods=["POST"])
def ejecutar():
    data = request.get_json(force=True)

    tamano = data.get("tamano", "chico")
    if tamano not in DATASETS:
        return jsonify({"error": f"Tamaño inválido: '{tamano}'. Opciones: chico, mediano, grande"}), 400

    ds = DATASETS[tamano]

    for clave in ("archivo_elementos", "archivo_restricciones"):
        if not os.path.exists(ds[clave]):
            return jsonify({"error": f"Archivo no encontrado: {ds[clave]}"}), 500

    params = {
        "ancho":                 ds["ancho"],
        "alto":                  ds["alto"],
        "archivo_elementos":     ds["archivo_elementos"],
        "archivo_restricciones": ds["archivo_restricciones"],
        "tam_poblacion": int(data.get("tam_poblacion", 30)),
        "p_cruza":       float(data.get("p_cruza",     0.70)),
        "p_mut_ind":     float(data.get("p_mut_ind",   0.30)),
        "p_mut_gen":     float(data.get("p_mut_gen",   0.25)),
        "generaciones":  int(data.get("generaciones",  200)),
        "wE": float(data.get("wE", 0.25)),
        "wF": float(data.get("wF", 0.30)),
        "wC": float(data.get("wC", 0.30)),
        "wP": float(data.get("wP", 0.15)),
    }

    try:
        resultado = ejecutar_ag(params)
        return jsonify(resultado)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(debug=True)