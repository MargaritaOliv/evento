"""
=============================================================
  EVENTOESPACIO — Back-end Flask
  Solo actúa como puente entre la web y el motor del AG.
  Toda la lógica del AG vive en eventoespacio_ag.py
=============================================================
"""

from flask import Flask, render_template, request, jsonify
from eventoespacio_ag import ejecutar_ag   # ← motor único

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/ejecutar", methods=["POST"])
def ejecutar():
    data = request.get_json()
    params = {
        "ancho":          int(data.get("ancho",          20)),
        "alto":           int(data.get("alto",           20)),
        "tam_poblacion":  int(data.get("tam_poblacion",  10)),
        "p_cruza":      float(data.get("p_cruza",       0.70)),
        "p_mut_ind":    float(data.get("p_mut_ind",     0.30)),
        "p_mut_gen":    float(data.get("p_mut_gen",     0.25)),
        "generaciones":   int(data.get("generaciones",  200)),
    }
    resultado = ejecutar_ag(params)
    return jsonify(resultado)


if __name__ == "__main__":
    app.run(debug=True)