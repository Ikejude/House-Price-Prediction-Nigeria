from flask import Flask, render_template, request
import numpy as np
import joblib  # Or use pickle if you have saved the model with pickle

app = Flask(__name__)

# Load the pre-trained model (Make sure you have the correct model file)
model = joblib.load('Model_svr.pkl')  # Replace with your model filename

# Define the list of towns
towns = [
    "Aba", "Abeokuta North", "Abeokuta South", "Abraka", "Ado-Ekiti", "Ado-Odo/Ota", "Afijio", "Agbara",
    "Agbara-Igbesa", "Agege", "Ajah", "Akinyele", "Akure", "Alimosho", "Amuwo Odofin", "Aniocha South", "Apapa",
    "Apo", "Arepo", "Asaba", "Asokoro District", "Ayobo", "Badagry", "Bwari", "Calabar", "Central Business District",
    "Chikun", "Dakibiyu", "Dakwo", "Danja", "Dape", "Dei-Dei", "Dekina", "Diplomatic Zones", "Duboyi", "Durumi",
    "Dutse", "Ede South", "Egbe", "Egbeda", "Egor", "Ejigbo", "Eket", "Eko Atlantic City", "Eleme", "Enugu", "Epe",
    "Ethiope West", "Ewekoro", "Gaduwa", "Galadimawa", "Garki", "Gbagada", "Gudu", "Guzamala", "Guzape District",
    "Gwagwalada", "Gwarinpa", "Ibadan", "Ibadan North", "Ibadan North-East", "Ibadan North-West", "Ibadan South-West",
    "Ibafo", "Ibarapa North", "Ibeju", "Ibeju Lekki", "Idimu", "Ido", "Idu Industrial", "Ifako-Ijaiye", "Ifo",
    "Ijaiye", "Ijebu Ode", "Ijede", "Ijesha", "Ijoko", "Ikeja", "Ikorodu", "Ikot Ekpene", "Ikotun", "Ikoyi",
    "Ikpoba Okha", "Ikwerre", "Ilorin East", "Ilorin South", "Ilorin West", "Ilupeju", "Imota", "Ipaja", "Isheri",
    "Isheri North", "Isolo", "Jabi", "Jahi", "Jikwoyi", "Jos North", "Jos South", "KM 46", "Kabusa", "Kado",
    "Kaduna North", "Kaduna South", "Kafe", "Kagini", "Kano", "Karmo", "Karsana", "Karshi", "Karu", "Katampe",
    "Kaura", "Keffi", "Ketu", "Kosofe", "Kubwa", "Kuje", "Kukwaba", "Kurudu", "Kusada", "Kyami", "Lagos Island",
    "Lekki", "Life Camp", "Lokogoma District", "Lokoja", "Lugbe District", "Mabushi", "Magboro", "Magodo", "Maitama District",
    "Mararaba", "Maryland", "Mbora (Nbora)", "Mowe Ofada", "Mowe Town", "Mpape", "Mushin", "Nasarawa", "Nassarawa",
    "Nyanya", "Obafemi Owode", "Obio-Akpor", "Ogijo", "Ogudu", "Ohaji/Egbema", "Ojo", "Ojodu", "Ojota", "Oke-Aro",
    "Oke-Odo", "Okene", "Okpe", "Oluyole", "Oredo", "Orile", "Orozo", "Oshodi", "Osogbo", "Ovia North-East",
    "Owerri Municipal", "Owerri North", "Owerri West", "Oyigbo", "Oyo West", "Paikoro", "Port Harcourt", "Sagamu",
    "Sango Ota", "Shomolu", "Simawa", "Surulere", "Udu", "Ughelli North", "Ughelli South", "Uhunmwonde", "Umuahia",
    "Utako", "Uvwie", "Uyo", "Victoria Island (VI)", "Warri", "Wumba", "Wuse", "Wuse 2", "Wuye", "Yaba", "Yenagoa",
    "Yewa South"
]

# Define the prediction function
@app.route('/')
def home():
    return render_template('index.html')
# @app.route('/')
# def home():
#     return render_template('index.html', towns=towns)

@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    toilets = int(request.form['toilets'])
    parking_space = int(request.form['parking_space'])
    title = int(request.form['title'])
    town = int(request.form['town'])
    state = int(request.form['state'])
        
    # Put input into numpy array
    features = np.array([[bedrooms, bathrooms, toilets, parking_space, title, town, state]])
        
    # Predict house price
    predicted_price = model.predict(features)[0]

    return render_template('index.html', predicted_price=f"₦ {predicted_price:,.2f}")

if __name__ == '__main__':
    app.run(debug=True)