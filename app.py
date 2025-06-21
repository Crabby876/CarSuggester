from flask import Flask, render_template, request, redirect, url_for, flash
from pymongo import MongoClient
import joblib
import pandas as pd
import os

_model = None
_columns = None

def get_model():
    global _model, _columns
    if _model is None or _columns is None:
        _model, _columns = joblib.load("modell_komprimiert.joblib")
    return _model, _columns

app = Flask(__name__)
app.secret_key = "123456789"
modell_datei = "modell_komprimiert.joblib"

if os.path.exists(modell_datei):
    RF, spalten = get_model()

getriebe_typen = ['MANUAL', 'AUTOMATIC', 'AUTOMATED_MANUAL']
antrieb_typen = ['rear wheel drive', 'front wheel drive', 'all wheel drive']
karosserie_typen = ['Limousine', 'Cabrio', 'Combi', 'Coupe', 'SUV', 'Van', 'Pickup']

letzte_daten = {}

@app.route('/', methods=['GET', 'POST'])
def index():
    global RF, spalten, letzte_daten
    vorhersage = None
    feedback = None

    if request.method == 'POST':
        if 'zufrieden' in request.form:
            return redirect(url_for('index'))

        else:
            jahr = int(request.form['jahr'])
            ps = float(request.form['ps'])
            preis = float(request.form['preis']) * 1.19
            getriebe = request.form['getriebe']
            antrieb = request.form['antrieb']
            tueren = int(request.form['tueren'])
            karosserie = request.form['karosserie']

            getriebe_map = {'MANUAL': 1, 'AUTOMATIC': 0, 'AUTOMATED_MANUAL': -1}
            antrieb_map = {'rear wheel drive': 1, 'front wheel drive': 0, 'all wheel drive': -1}

            neue_daten = {
                'Year': jahr,
                'Engine HP': ps,
                'Transmission Type': getriebe_map.get(getriebe),
                'Driven_Wheels': antrieb_map.get(antrieb),
                'Number of Doors': tueren,
                'Price': preis,
                'Vehicle Style': karosserie,
            }
            letzte_daten = neue_daten.copy()

            input_df = pd.DataFrame([neue_daten])
            input_df = pd.get_dummies(input_df)

            for spalte in spalten:
                if spalte not in input_df.columns:
                    input_df[spalte] = 0
            input_df = input_df[spalten]

            vorhersage = RF.predict(input_df)[0]

    return render_template("index.html", getriebe_typen=getriebe_typen, antrieb_typen=antrieb_typen, karosserie_typen=karosserie_typen, vorhersage=vorhersage, feedback=feedback)

if __name__ == "__main__":
    app.run(debug=False)
