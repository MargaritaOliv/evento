"""
=============================================================
  EVENTOESPACIO — Back-end Flask  (versión actualizada)
  Cambios:
    · Acepta archivos CSV subidos desde el frontend (dataset)
    · Devuelve hist_peor para gráfica completa
    · Devuelve top3 (los 3 mejores individuos)
=============================================================
"""

import os
import io
import tempfile
from flask import Flask, render_template, request, jsonify
from eventoespacio_ag import ejecutar_ag

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ejecutar", methods=["POST"])
def ejecutar():
    """
    Acepta multipart/form-data con:
      · campos de texto: ancho, alto, tam_poblacion, etc.
      · archivos opcionales: elementos_csv, restricciones_csv
    """
    # ── Parámetros del AG ──────────────────────────────────
    def geti(k, d): return int(request.form.get(k, d))
    def getf(k, d): return float(request.form.get(k, d))

    params = {
        "ancho":         geti("ancho",         20),
        "alto":          geti("alto",          20),
        "tam_poblacion": geti("tam_poblacion", 10),
        "p_cruza":       getf("p_cruza",       0.70),
        "p_mut_ind":     getf("p_mut_ind",     0.30),
        "p_mut_gen":     getf("p_mut_gen",     0.25),
        "generaciones":  geti("generaciones",  200),
    }

    # ── Archivos CSV (opcionales) ──────────────────────────
    archivos = {}
    for campo in ("elementos_csv", "restricciones_csv"):
        f = request.files.get(campo)
        if f and f.filename:
            # Guarda en un temporal y pasa la ruta al motor
            tmp = tempfile.NamedTemporaryFile(
                delete=False, suffix=".csv",
                mode="wb"
            )
            f.save(tmp.name)
            archivos[campo] = tmp.name

    # ── Preset de dataset (si viene preset, usa los CSV incluidos) ──
    preset = request.form.get("preset")
    if preset and preset in ("pequeno", "mediano", "grande"):
        preset_dir = os.path.join(BASE_DIR, "data", "datasets", preset)
        params["archivo_elementos"]     = os.path.join(preset_dir, "catalogo_elementos.csv")
        params["archivo_restricciones"] = os.path.join(preset_dir, "catalogo_restricciones.csv")
    else:
        params["archivo_elementos"]     = archivos.get("elementos_csv")
        params["archivo_restricciones"] = archivos.get("restricciones_csv")

    resultado = ejecutar_ag(params)

    # Limpia temporales
    for ruta in archivos.values():
        try:
            os.remove(ruta)
        except Exception:
            pass

    return jsonify(resultado)


if __name__ == "__main__":
    app.run(debug=True)