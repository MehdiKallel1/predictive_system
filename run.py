from flask import Flask, render_template, jsonify
import pandas as pd
import os
from datetime import datetime

app = Flask(__name__, 
            template_folder='templates',
            static_folder='static')

# Function to load and process macroeconomic data
def load_macro_data():
    # Use the specified path to your macro data
    data_path = r'C:\Users\MK 10\OneDrive\Bureau\Stage PFE\DataFrames\Merged_df\monthly_data.csv'
    
    # If the file doesn't exist, inform user
    if not os.path.exists(data_path):
        print(f"Error: Could not find the CSV file at: {data_path}")
        # Return empty dataframe with the expected columns
        return pd.DataFrame(columns=['Date', 'is_predicted', 'Credit_Interieur', 'Impots_Revenus', 
                                    'Inflation_Rate', 'Paiements_Interet', 'Taux_Interet', 
                                    'RNB_Par_Habitant', 'Masse_Monetaire', 'PIB_US_Courants', 
                                    'RNB_US_Courants'])
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Convert date column to datetime (handle potential format issues)
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception as e:
        print(f"Error converting date column: {e}")
        print("Attempting alternative date format...")
        try:
            # Try with different format in case the format is different
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        except:
            print("Could not convert date column. Check the date format in your CSV.")
    
    # Create a flag for predicted data (2025-2026)
    df['is_predicted'] = df['Date'].dt.year >= 2025
    
    return df

@app.route('/')
def index():
    # Get the list of available indicators
    df = load_macro_data()
    indicators = [col for col in df.columns if col not in ['Date', 'is_predicted']]
    
    # Print information about the data for debugging
    print(f"Data loaded successfully. Found {len(df)} rows and {len(indicators)} indicators.")
    print(f"Available indicators: {indicators}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"Predicted data points: {df['is_predicted'].sum()}")
    
    return render_template('index.html', indicators=indicators)

@app.route('/data/<indicator>')
def get_data(indicator):
    df = load_macro_data()
    
    # Make sure the indicator exists in the dataframe
    if indicator not in df.columns:
        return jsonify({'error': 'Indicator not found'}), 404
    
    # Make a copy to avoid modifying the original dataframe
    chart_df = df.copy()
    
    # Handle NaN values
    chart_df = chart_df.dropna(subset=[indicator])
    
    # Sort by date to ensure correct ordering
    chart_df = chart_df.sort_values('Date')
    
    # Prepare data for the chart
    data = {
        'dates': chart_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'values': chart_df[indicator].tolist(),
        'is_predicted': chart_df['is_predicted'].tolist()
    }
    
    return jsonify(data)

@app.route('/summary')
def get_summary():
    df = load_macro_data()
    
    # Filter for 2025-2026 predictions
    predictions = df[df['is_predicted']]
    
    # Calculate basic statistics for the predictions
    indicators = [col for col in df.columns if col not in ['Date', 'is_predicted']]
    
    summary = {}
    for indicator in indicators:
        indicator_data = predictions[indicator]
        if not indicator_data.empty:
            summary[indicator] = {
                'min': float(indicator_data.min()),
                'max': float(indicator_data.max()),
                'mean': float(indicator_data.mean()),
                'start': float(indicator_data.iloc[0]),
                'end': float(indicator_data.iloc[-1]),
                'change_percent': float((indicator_data.iloc[-1] - indicator_data.iloc[0]) / indicator_data.iloc[0] * 100)
            }
        else:
            summary[indicator] = {
                'min': 0,
                'max': 0,
                'mean': 0,
                'start': 0,
                'end': 0,
                'change_percent': 0
            }
    
    return jsonify(summary)

if __name__ == '__main__':
    app.run(debug=True)