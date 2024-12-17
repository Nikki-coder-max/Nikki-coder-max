from flask import Flask, render_template, request
import joblib
import pandas as pd

# Initialize Flask app
app = Flask(__name__)

# Load the trained model pipeline
model_path = r'C:\Users\user\Downloads\archive\flight_fare_prediction_pipeline.pkl'
model_pipeline = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('flight_html_app.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Collect form data and handle missing values
            airline = request.form['airline']
            source = request.form['source']
            destination = request.form['destination']
            date_of_journey = request.form['date_of_journey']
            dep_time = request.form['dep_time']
            arrival_time = request.form['arrival_time']
            duration = request.form['duration']
            total_stops = request.form['total_stops']
            additional_info = request.form['additional_info']

            # Default to 0 if 'duration' or 'total_stops' are empty
            duration = int(duration) if duration.strip() else 0
            total_stops = int(total_stops) if total_stops.strip() else 0

            # Preprocess input data
            input_data = pd.DataFrame([{
                'Airline': airline,
                'Source': source,
                'Destination': destination,
                'Day': pd.to_datetime(date_of_journey).day,
                'Month': pd.to_datetime(date_of_journey).month,
                'Dep_Hour': pd.to_datetime(dep_time).hour,
                'Dep_Minute': pd.to_datetime(dep_time).minute,
                'Arrival_Hour': pd.to_datetime(arrival_time).hour,
                'Arrival_Minute': pd.to_datetime(arrival_time).minute,
                'Duration': duration,
                'Total_Stops': total_stops,
                'Additional_Info': additional_info
            }])

            # Predict flight fare
            prediction = model_pipeline.predict(input_data)[0]
            result = f"The predicted flight fare is ₹{format(prediction, '.2f')}"
        except ValueError as ve:
            result = f"Error: Please ensure all numeric fields are valid numbers."
        except Exception as e:
            result = f"Error in prediction: {e}"

        return render_template('flight_html_app.html', prediction=result)
    
if __name__ == "__main__":
    app.run(debug=True)

