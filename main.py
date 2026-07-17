from flask import Flask, render_template, request
import joblib
#import numpy as np

app = Flask(__name__)

model = joblib.load("RF_pca95.joblib")
scaler = joblib.load("scaler.joblib")
pca = joblib.load("pca95.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")    
    
@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method=='POST':
        features = [float(x) for x in request.form.values()]

        #features = np.array(features).reshape(1, -1)
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        prediction = model.predict(features_pca)[0]
        
        return render_template("predict.html", prediction=prediction)
        
    return render_template("predict.html")
    
if __name__=="__main__":
    app.run(debug=True)