from flask import Flask, render_template, request, jsonify
import os
import traceback
from eventoespacio_ag import run_ag, AG_CONFIG

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Dimensiones correctas para cada tamaño de venue
DATASETS = {
    "chico": {
        "venue_ancho": 50,
        "venue_alto":  40,
        "elementos":     os.path.join(BASE_DIR, "data", "elementos",     "chico.csv"),
        "restricciones": os.path.join(BASE_DIR, "data", "restricciones", "chico.csv"),
    },
    "mediano": {
        "venue_ancho": 80,
        "venue_alto":  60,
        "elementos":     os.path.join(BASE_DIR, "data", "elementos",     "mediano.csv"),
        "restricciones": os.path.join(BASE_DIR, "data", "restricciones", "mediano.csv"),
    },
    "grande": {
        "venue_ancho": 120,
        "venue_alto":  90,
        "elementos":     os.path.join(BASE_DIR, "data", "elementos",     "grande.csv"),
        "restricciones": os.path.join(BASE_DIR, "data", "restricciones", "grande.csv"),
    },
}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json()

    tamano       = data.get("tamano", "chico")
    generaciones = int(data.get("generaciones", 200))
    poblacion    = int(data.get("poblacion", 80))

    if tamano not in DATASETS:
        return jsonify({"ok": False, "error": f"Tamaño inválido: {tamano}"}), 400

    ds = DATASETS[tamano]

    # Verificar que los archivos existen
    for ruta in (ds["elementos"], ds["restricciones"]):
        if not os.path.exists(ruta):
            return jsonify({
                "ok": False,
                "error": f"Archivo no encontrado: {ruta}"
            }), 500

    config = {
        **AG_CONFIG,
        "GENERACIONES": generaciones,
        "POBLACION":    poblacion,
    }

    try:
        resultado = run_ag(
            venue_ancho=ds["venue_ancho"],
            venue_alto=ds["venue_alto"],
            ruta_elementos=ds["elementos"],
            ruta_restricciones=ds["restricciones"],
            config=config,
        )
        return jsonify({"ok": True, "data": resultado})

    except Exception:
        error_detalle = traceback.format_exc()
        print(error_detalle)
        return jsonify({"ok": False, "error": error_detalle}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)