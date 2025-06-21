import joblib

# Original laden (besteht aus Tuple: (modell, spalten))
modell, spalten = joblib.load('modell_rf.pkl')

# Komprimiert speichern
joblib.dump((modell, spalten), 'modell_komprimiert.joblib', compress=("xz",9))

print("Modell und Spaltenliste wurden komprimiert als 'modell_komprimiert.joblib' gespeichert.")