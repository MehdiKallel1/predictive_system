from flask import Blueprint, render_template, jsonify
import pandas as pd
import os
import json
from datetime import datetime

main_bp = Blueprint('main', __name__)

# Function to load and process macroeconomic data
def load_macro_data():
    # Change this path to the location of your CSV file
    data_path = r'C:\Users\MK 10\OneDrive\Bureau\Stage PFE\DataFrames\Merged_df\monthly_data.csv'
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Convert date column to datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Create a flag for predicted data (2025-2026)
    df['is_predicted'] = df['Date'].dt.year >= 2025
    
    return df

@main_bp.route('/')
def index():
    # Get the list of available indicators
    df = load_macro_data()
    indicators = [col for col in df.columns if col not in ['Date', 'is_predicted']]
    
    return render_template('index.html', indicators=indicators)

@main_bp.route('/data/<indicator>')
def get_data(indicator):
    df = load_macro_data()
    
    # Make sure the indicator exists in the dataframe
    if indicator not in df.columns:
        return jsonify({'error': 'Indicator not found'}), 404
    
    # Prepare data for the chart
    data = {
        'dates': df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'values': df[indicator].tolist(),
        'is_predicted': df['is_predicted'].tolist()
    }
    
    return jsonify(data)

@main_bp.route('/summary')
def get_summary():
    df = load_macro_data()
    
    # Filter for 2025-2026 predictions
    predictions = df[df['is_predicted']]
    
    # Calculate basic statistics for the predictions
    indicators = [col for col in df.columns if col not in ['Date', 'is_predicted']]
    
    summary = {}
    for indicator in indicators:
        indicator_data = predictions[indicator]
        summary[indicator] = {
            'min': float(indicator_data.min()),
            'max': float(indicator_data.max()),
            'mean': float(indicator_data.mean()),
            'start': float(indicator_data.iloc[0]),
            'end': float(indicator_data.iloc[-1]),
            'change_percent': float((indicator_data.iloc[-1] - indicator_data.iloc[0]) / indicator_data.iloc[0] * 100)
        }
    
    return jsonify(summary)