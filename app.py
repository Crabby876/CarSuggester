from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Flask App Setup
app = Flask(__name__)
app.secret_key = "123456789"

# Dateien
modell_datei = "modell_rf.pkl"
daten_datei = "daten.json"


getriebe_typen = ['MANUAL', 'AUTOMATIC', 'AUTOMATED_MANUAL']
antrieb_typen = ['rear wheel drive', 'front wheel drive', 'all wheel drive']
karosserie_typen = ['Limousine', 'Cabrio', 'Combi', 'Coupe', 'SUV', 'Van', 'Pickup']


letzte_daten = {}

# Funktion: Daten laden und trainieren
def daten_vorbereiten_und_trainieren(daten, modell_datei="modell_rf.pkl"):
    df = pd.DataFrame(daten)

    df['Transmission Type'] = df['Transmission Type'].replace({
        'MANUAL': 1, 'AUTOMATIC': 0, 'AUTOMATED_MANUAL': -1, 'DIRECT_DRIVE': 0, 'UNKNOWN': None})
    df['Driven_Wheels'] = df['Driven_Wheels'].replace({
        'rear wheel drive': 1, 'front wheel drive': 0, 'all wheel drive': -1, 'four wheel drive': -1})
    df['Vehicle Style'] = df['Vehicle Style'].replace({
        'Sedan': 'Limousine', 'Convertible': 'Cabrio', 'Convertible SUV': 'Cabrio',
        '2dr Hatchback': 'Combi', '4dr Hatchback': 'Combi', 'Wagon': 'Combi',
        'Coupe': 'Coupe', '4dr SUV': 'SUV', '2dr SUV': 'SUV',
        'Passenger Van': 'Van', 'Passenger Minivan': 'Van', 'Cargo Van': 'Van',
        'Cargo Minivan': 'Van', 'Crew Cab Pickup': 'Pickup', 'Regular Cab Pickup': 'Pickup',
        'Extended Cab Pickup': 'Pickup'
    })

    df['Engine HP'] = pd.to_numeric(df['Engine HP'], errors='coerce')
    df['Make'] = df['Make'].astype(str)
    df['Model'] = df['Model'].astype(str)
    df['MarkeModell'] = df['Make'] + " " + df['Model']
    df = df.dropna()

    y = df['MarkeModell']
    df_encoded = pd.get_dummies(df, columns=['Vehicle Style', 'Make', 'Model'])

    X = df_encoded.drop(columns=['MarkeModell', 'city mpg', 'Popularity', 'highway MPG',
                                 'Vehicle Size', 'Market Category', 'Engine Cylinders',
                                 'Engine Fuel Type'], errors='ignore')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    RF = RandomForestClassifier()
    RF.fit(X_train, y_train)

    joblib.dump((RF, X.columns.tolist()), modell_datei)

    return RF, X.columns.tolist()


# Modell laden oder trainieren
if os.path.exists(modell_datei):
    RF, spalten = joblib.load(modell_datei)
else:
    with open(daten_datei, "r", encoding="utf-8") as f:
        daten = json.load(f)
    RF, spalten = daten_vorbereiten_und_trainieren(daten, modell_datei)


# Flask Routes
@app.route('/', methods=['GET', 'POST'])
def index():
    global RF, spalten, letzte_daten
    vorhersage = None

    if request.method == 'POST':
        # Prüfen, ob Zurück-Button gedrückt wurde
        if 'zufrieden' in request.form:
            return redirect(url_for('index'))

        # Sonst Eingabedaten verarbeiten
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
                input_df[spalte] = None
        input_df = input_df[spalten]

        vorhersage = RF.predict(input_df)[0]

    return render_template(
        "index.html",
        getriebe_typen=getriebe_typen,
        antrieb_typen=antrieb_typen,
        karosserie_typen=karosserie_typen,
        vorhersage=vorhersage
    )



if __name__ == "__main__":
    app.run(debug=True)
