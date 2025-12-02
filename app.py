# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('mobile_price_model.pkl', 'rb') as f:
    saved_data = pickle.load(f)

model = saved_data['model']
encoders = saved_data['encoders']
feature_names = saved_data['feature_names']

@app.route('/')
def index():
    # Dropdown options bhejne ke liye
    options = {}
    for col, le in encoders.items():
        options[col] = le.classes_
            
    return render_template('index.html', options=options, feature_names=feature_names)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = []
        
        for feature in feature_names:
            user_input = request.form.get(feature)
            
            # Encode categorical inputs
            if feature in encoders:
                le = encoders[feature]
                if user_input in le.classes_:
                    val = le.transform([user_input])[0]
                else:
                    val = 0 # Fallback
            else:
                # Numeric features
                val = float(user_input)
            
            input_data.append(val)
        
        # Predict Price
        prediction_price = model.predict([input_data])[0]
        
        # Formatting result as currency
        result_text = f"Predicted Price: ₹ {int(prediction_price):,}"
        
        return render_template('index.html', 
                               prediction_text=result_text,
                               options={col: le.classes_ for col, le in encoders.items()},
                               feature_names=feature_names)
                               
    except Exception as e:
        return render_template('index.html', 
                               prediction_text=f"Error: {str(e)}",
                               options={col: le.classes_ for col, le in encoders.items()},
                               feature_names=feature_names)

if __name__ == '__main__':
    app.run(debug=True)