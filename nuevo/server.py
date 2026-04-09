from flask import Flask, render_template, request, jsonify
import os
import traceback
# Importamos la configuración desde tu algoritmo
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
    # PASO CLAVE: Le enviamos AG_CONFIG al HTML (Jinja2) para no tener datos hardcodeados
    return render_template("index.html", config=AG_CONFIG)

@app.route("/api/run", methods=["POST"])
def api_run():
    data = request.get_json()

    # Recibimos el tamaño desde el frontend
    tamano       = data.get("tamano", "chico")
    
    # Si el frontend nos manda generaciones/población, las usamos. Si no, usamos las de AG_CONFIG
    generaciones = int(data.get("generaciones", AG_CONFIG.get("GENERACIONES", 200)))
    poblacion    = int(data.get("poblacion", AG_CONFIG.get("POBLACION", 80)))

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

    # Sobreescribimos la configuración por defecto con lo que puso el usuario en la interfaz
    config_actualizada = {
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
            config=config_actualizada,
        )
        return jsonify({"ok": True, "data": resultado})

    except Exception:
        error_detalle = traceback.format_exc()
        print(error_detalle)
        return jsonify({"ok": False, "error": error_detalle}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)