# CarSuggester

## Projekt Beschreibung  

Der **CarSuggester** ist eine intelligente Webanwendung, die Nutzern hilft, das passende Auto zu finden.  
Das Programm basiert auf einem **Random Forest Machine Learning Modell** und einer Datenbank mit realen Fahrzeugen.  

Der Nutzer gibt seine Präferenzen ein (z. B. **Getriebeart, Antrieb, Karosserie-Typ**) und erhält anschließend eine **Vorhersage für ein passendes Auto (Marke & Modell)**.  

---

## Technologien  

- **Python**  
- **Flask** – Web-Framework  
- **scikit-learn** – Machine Learning (Random Forest Classifier)  
- **pandas** – Datenverarbeitung  
- **joblib** – Modell speichern/laden  
- **JSON** – Datenspeicher für Fahrzeugdaten  

---

## Features  

- Benutzerfreundliche **Web-Oberfläche** (Flask)  
- **Eingabefelder** für Fahrzeugpräferenzen (z. B. Getriebe, Antrieb, Karosserie)  
- **Random Forest Modell** trifft Vorhersagen auf Basis historischer Daten  
- Ausgabe: **Marke & Modell** als Empfehlung  
- Möglichkeit, neue Daten einzulesen und das Modell neu zu trainieren  
