from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipelines import CustomData, PredictPipeline

application = Flask(__name__)

app = application

#Route for a home page
@app.route('/')
def index():
    return render_template('index.html')

#Route for a home page
@app.route('/predictdata', methods = ['GET','POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        data = CustomData(
            hotel=request.form.get('hotel'),
            arrival_date_month=request.form.get('arrival_date_month'),
            meal=request.form.get('meal'),
            country=request.form.get('country'),
            market_segment=request.form.get('market_segment'),
            customer_type=request.form.get('customer_type'),
            lead_time=request.form.get('lead_time'),
            is_repeated_guest=request.form.get('is_repeated_guest'),
            previous_cancellations=request.form.get('previous_cancellations'),
            previous_bookings_not_canceled=request.form.get('previous_bookings_not_canceled'),
            booking_changes=request.form.get('booking_changes'),
            days_in_waiting_list=request.form.get('days_in_waiting_list'),
            adr=request.form.get('adr'),
            required_car_parking_spaces=request.form.get('required_car_parking_spaces'),
            total_of_special_requests=request.form.get('total_of_special_requests'),
            total_guests=request.form.get('total_guests'),
            total_stay_length=request.form.get('total_stay_length'),
            is_family=request.form.get('is_family'),
            is_deposit_given=request.form.get('is_deposit_given'),
            is_room_upgraded=request.form.get('is_room_upgraded')
        )
        
        pred_df = data.get_data_as_dataframe()
        print(pred_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        return render_template("home.html", results=results[0])
        

if __name__=="__main__":
    app.run(host="0.0.0.0", debug=True)
        
        