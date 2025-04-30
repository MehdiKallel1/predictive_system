from flask import Blueprint, render_template, jsonify, current_app
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime

main_bp = Blueprint('main', __name__)

# Function to load and process macroeconomic data
def load_macro_data():
    # Use relative path for data files
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'monthly_data.csv')
    
    # Check if file exists
    if not os.path.exists(data_path):
        print(f"Error: Could not find the CSV file at: {data_path}")
        # Create a data directory if it doesn't exist
        os.makedirs(os.path.dirname(data_path), exist_ok=True)
        
        # Return empty dataframe with the expected columns
        dummy_df = pd.DataFrame(columns=['Date', 'is_predicted', 'Credit_Interieur', 'Impots_Revenus', 
                                    'Inflation_Rate', 'Paiements_Interet', 'Taux_Interet', 
                                    'RNB_Par_Habitant', 'Masse_Monetaire', 'PIB_US_Courants', 
                                    'RNB_US_Courants'])
        
        # Add some sample data for testing
        dates = pd.date_range(start='2020-01-01', end='2026-12-31', freq='MS')
        dummy_df['Date'] = dates
        dummy_df['is_predicted'] = dummy_df['Date'].dt.year >= 2025
        
        # Generate some random data for each indicator
        for col in dummy_df.columns:
            if col not in ['Date', 'is_predicted']:
                base_value = 100
                dummy_df[col] = [base_value + i * 2 + np.random.normal(0, 5) for i in range(len(dates))]
        
        # Save the dummy data for future use
        dummy_df.to_csv(data_path, index=False)
        return dummy_df
    
    # Load the data
    try:
        df = pd.read_csv(data_path)
        
        # Convert date column to datetime
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create a flag for predicted data (2025-2026)
        df['is_predicted'] = df['Date'].dt.year >= 2025
        
        print(f"Macro data loaded successfully. Found {len(df)} rows.")
        print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"Predicted data points: {df['is_predicted'].sum()}")
        
    except Exception as e:
        print(f"Error loading macro data: {e}")
        df = pd.DataFrame(columns=['Date', 'is_predicted', 'Credit_Interieur', 'Impots_Revenus', 
                                  'Inflation_Rate', 'Paiements_Interet', 'Taux_Interet', 
                                  'RNB_Par_Habitant', 'Masse_Monetaire', 'PIB_US_Courants', 
                                  'RNB_US_Courants'])
    
    return df

# Function to load and process company data
def load_company_data():
    # Use relative paths for data files
    historical_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'synthetic_company_data.csv')
    predicted_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'predicted_company_data_2025_2026.csv')
    
    # Create data directory if it doesn't exist
    os.makedirs(os.path.dirname(historical_path), exist_ok=True)
    
    # Check if files exist
    if not os.path.exists(historical_path):
        print(f"Error: Could not find the CSV file at: {historical_path}")
        # Create dummy data for testing
        dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='MS')
        dummy_df = pd.DataFrame({
            'Date': dates,
            'Revenue': [100000 + i * 1000 + np.random.normal(0, 2000) for i in range(len(dates))],
            'Profit': [20000 + i * 200 + np.random.normal(0, 1000) for i in range(len(dates))],
            'Risk_Score': [50 + np.sin(i/6) * 10 for i in range(len(dates))],
            'is_predicted': False
        })
        dummy_df.to_csv(historical_path, index=False)
        
        # Create dummy predicted data
        pred_dates = pd.date_range(start='2025-01-01', end='2026-12-31', freq='MS')
        pred_df = pd.DataFrame({
            'Date': pred_dates,
            'Revenue': [dummy_df['Revenue'].iloc[-1] + i * 1200 + np.random.normal(0, 3000) for i in range(len(pred_dates))],
            'Profit': [dummy_df['Profit'].iloc[-1] + i * 250 + np.random.normal(0, 1500) for i in range(len(pred_dates))],
            'Risk_Score': [dummy_df['Risk_Score'].iloc[-1] + np.sin(i/6) * 15 for i in range(len(pred_dates))],
            'is_predicted': True
        })
        pred_df.to_csv(predicted_path, index=False)
        
        historical_df = dummy_df
        predicted_df = pred_df
    else:
        # Load historical data
        try:
            historical_df = pd.read_csv(historical_path)
            historical_df['Date'] = pd.to_datetime(historical_df['Date'])
            historical_df['is_predicted'] = False
        except Exception as e:
            print(f"Error loading historical company data: {e}")
            historical_df = pd.DataFrame(columns=['Date', 'Revenue', 'Profit', 'Risk_Score', 'is_predicted'])
        
        if not os.path.exists(predicted_path):
            print(f"Error: Could not find the CSV file at: {predicted_path}")
            # Create dummy predicted data
            if not historical_df.empty:
                pred_dates = pd.date_range(start='2025-01-01', end='2026-12-31', freq='MS')
                pred_df = pd.DataFrame({
                    'Date': pred_dates,
                    'Revenue': [historical_df['Revenue'].iloc[-1] + i * 1200 + np.random.normal(0, 3000) for i in range(len(pred_dates))],
                    'Profit': [historical_df['Profit'].iloc[-1] + i * 250 + np.random.normal(0, 1500) for i in range(len(pred_dates))],
                    'Risk_Score': [historical_df['Risk_Score'].iloc[-1] + np.sin(i/6) * 15 for i in range(len(pred_dates))],
                    'is_predicted': True
                })
                pred_df.to_csv(predicted_path, index=False)
                predicted_df = pred_df
            else:
                predicted_df = pd.DataFrame(columns=['Date', 'Revenue', 'Profit', 'Risk_Score', 'is_predicted'])
        else:
            # Load predicted data
            try:
                predicted_df = pd.read_csv(predicted_path)
                predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
                predicted_df['is_predicted'] = True
            except Exception as e:
                print(f"Error loading predicted company data: {e}")
                predicted_df = pd.DataFrame(columns=['Date', 'Revenue', 'Profit', 'Risk_Score', 'is_predicted'])
    
    # Combine the dataframes
    df = pd.concat([historical_df, predicted_df], ignore_index=True)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Print information about the data for debugging
    print(f"Company data loaded. Found {len(df)} rows.")
    print(f"Date range: {df['Date'].min() if not df.empty else 'N/A'} to {df['Date'].max() if not df.empty else 'N/A'}")
    print(f"Predicted data points: {df['is_predicted'].sum() if not df.empty else 0}")
    
    return df

# Function to calculate correlations
def calculate_correlations():
    try:
        macro_df = load_macro_data()
        company_df = load_company_data()
        
        # Merge dataframes on Date
        merged_df = pd.merge(macro_df, company_df, on='Date', suffixes=('_macro', '_company'))
        
        # Get company metrics and macro indicators
        company_metrics = ['Revenue', 'Profit', 'Risk_Score']
        macro_indicators = [col for col in macro_df.columns if col not in ['Date', 'is_predicted']]
        
        # Calculate correlations
        correlations = {}
        for metric in company_metrics:
            correlations[metric] = {}
            for indicator in macro_indicators:
                if metric in merged_df.columns and indicator in merged_df.columns:
                    correlation = merged_df[[metric, indicator]].corr().iloc[0, 1]
                    correlations[metric][indicator] = float(correlation) if not np.isnan(correlation) else 0
                else:
                    correlations[metric][indicator] = 0
        
        print("Correlations calculated successfully.")
        return correlations
    except Exception as e:
        print(f"Error calculating correlations: {e}")
        return {}

@main_bp.route('/')
def index():
    # Get the list of available indicators
    macro_df = load_macro_data()
    company_df = load_company_data()
    
    macro_indicators = [col for col in macro_df.columns if col not in ['Date', 'is_predicted']]
    company_metrics = ['Revenue', 'Profit', 'Risk_Score']
    
    return render_template('index.html', 
                          macro_indicators=macro_indicators,
                          company_metrics=company_metrics)

@main_bp.route('/macro-data/<indicator>')
def get_macro_data(indicator):
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

@main_bp.route('/company-data/<metric>')
def get_company_data(metric):
    df = load_company_data()
    
    # Make sure the metric exists in the dataframe
    if metric not in df.columns:
        return jsonify({'error': 'Metric not found'}), 404
    
    # Handle NaN values
    chart_df = df.dropna(subset=[metric])
    
    # Sort by date to ensure correct ordering
    chart_df = chart_df.sort_values('Date')
    
    # Prepare data for the chart
    data = {
        'dates': chart_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
        'values': chart_df[metric].tolist(),
        'is_predicted': chart_df['is_predicted'].tolist()
    }
    
    return jsonify(data)

@main_bp.route('/macro-summary')
def get_macro_summary():
    df = load_macro_data()
    
    # Filter for 2025-2026 predictions
    predictions = df[df['is_predicted']]
    
    # Calculate basic statistics for the predictions
    indicators = [col for col in df.columns if col not in ['Date', 'is_predicted']]
    
    summary = {}
    for indicator in indicators:
        if indicator in predictions.columns:
            indicator_data = predictions[indicator].dropna()  # Drop NaN values
            
            if not indicator_data.empty:
                try:
                    first_value = indicator_data.iloc[0]
                    last_value = indicator_data.iloc[-1]
                    change_percent = ((last_value - first_value) / first_value * 100)
                    
                    summary[indicator] = {
                        'min': float(indicator_data.min()),
                        'max': float(indicator_data.max()),
                        'mean': float(indicator_data.mean()),
                        'start': float(first_value),
                        'end': float(last_value),
                        'change_percent': float(change_percent)
                    }
                except Exception as e:
                    print(f"Error processing {indicator}: {e}")
                    summary[indicator] = {
                        'min': 0,
                        'max': 0,
                        'mean': 0,
                        'start': 0,
                        'end': 0,
                        'change_percent': 0
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

@main_bp.route('/company-summary')
def get_company_summary():
    df = load_company_data()
    
    # Filter for 2025-2026 predictions
    predictions = df[df['is_predicted']]
    
    # Calculate basic statistics for the predictions
    metrics = ['Revenue', 'Profit', 'Risk_Score']
    
    summary = {}
    for metric in metrics:
        if metric in predictions.columns:
            metric_data = predictions[metric].dropna()  # Drop NaN values
            
            if not metric_data.empty:
                try:
                    first_value = metric_data.iloc[0]
                    last_value = metric_data.iloc[-1]
                    change_percent = ((last_value - first_value) / first_value * 100)
                    
                    summary[metric] = {
                        'min': float(metric_data.min()),
                        'max': float(metric_data.max()),
                        'mean': float(metric_data.mean()),
                        'start': float(first_value),
                        'end': float(last_value),
                        'change_percent': float(change_percent)
                    }
                except Exception as e:
                    print(f"Error processing {metric}: {e}")
                    summary[metric] = {
                        'min': 0,
                        'max': 0,
                        'mean': 0,
                        'start': 0,
                        'end': 0,
                        'change_percent': 0
                    }
            else:
                summary[metric] = {
                    'min': 0,
                    'max': 0,
                    'mean': 0,
                    'start': 0,
                    'end': 0,
                    'change_percent': 0
                }
        else:
            summary[metric] = {
                'min': 0,
                'max': 0,
                'mean': 0,
                'start': 0,
                'end': 0,
                'change_percent': 0
            }
    
    return jsonify(summary)

@main_bp.route('/correlations')
def get_correlations():
    correlations = calculate_correlations()
    return jsonify(correlations)

# Add a route handler for any missing static files to provide better error messages
@main_bp.route('/static/<path:filename>')
def static_files(filename):
    try:
        return current_app.send_static_file(filename)
    except Exception as e:
        return jsonify({'error': f'Static file not found: {filename}'}), 404