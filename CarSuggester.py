from pymongo import MongoClient
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib


def daten_vorbereiten_und_trainieren(collection, modell_datei="modell_rf.pkl"):
    daten = list(collection.find())
    for d in daten:
        d.pop('_id', None)

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
