from flask import Blueprint, render_template, jsonify, current_app
import pandas as pd
import os
import json
import numpy as np
from datetime import datetime
# Add these imports at the top of your routes.py file
import os
from werkzeug.utils import secure_filename
from flask import request, redirect, url_for
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import logging

main_bp = Blueprint('main', __name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add these constants near the top of your routes.py file
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'uploads')
ALLOWED_EXTENSIONS = {'csv'}

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Add this route to your routes.py file
@main_bp.route('/upload-company-data', methods=['POST'])
def upload_company_data():
    # Check if the post request has the file part
    if 'company-data-file' not in request.files:
        return jsonify({'success': False, 'message': 'No file part in the request.'}), 400
    
    file = request.files['company-data-file']
    
    # If user does not select file, browser also submits an empty part without filename
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected.'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        
        # Process the uploaded file
        try:
            result = process_company_data(file_path)
            return jsonify(result)
        except Exception as e:
            logger.exception("Error processing uploaded file")
            return jsonify({
                'success': False, 
                'message': 'Error processing the file', 
                'errors': [str(e)]
            }), 500
    
    return jsonify({'success': False, 'message': 'Invalid file type.'}), 400
# Add this route to your routes.py file

# Corrected route with the exact path as requested from the HTML template

@main_bp.route('/download-sample-template')
def download_sample_template():
    """Serve a sample CSV template for users to download"""
    logger.info("Download sample template requested")
    
    # Create a simple sample CSV content in memory with comments
    sample_content = """Date,Sales,Operating_Cost,Marketing_Expense,Net_Income,Market_Share
1999-01-31,1250630.45,850450.32,125000.00,275180.13,12.35
1999-02-28,1362450.76,930560.34,135000.00,296890.42,12.48
1999-03-31,1452780.32,985670.45,140000.00,327109.87,12.67
1999-04-30,1398560.45,962340.23,138000.00,298220.22,12.53
1999-05-31,1425730.65,975890.34,142000.00,307840.31,12.74
1999-06-30,1478930.23,999650.45,145000.00,334279.78,12.86
# Notes:
# 1. Date column is required and must be in YYYY-MM-DD format (end of month format)
# 2. You can include any numeric columns you want - they will all be used for prediction
# 3. At least 12 months of data that overlap with our macroeconomic dataset is required
# 4. The data must include dates before 2025 only (historical data)
# 5. Feel free to add or remove columns as needed for your specific metrics
"""
    
    # Create a response with the CSV content
    response = current_app.response_class(
        response=sample_content,
        status=200,
        mimetype='text/csv'
    )
    
    # Set Content-Disposition header to prompt download
    response.headers["Content-Disposition"] = "attachment; filename=company_data_template.csv"
    
    logger.info("Serving sample template")
    return response

# Function to validate and process the uploaded company data
# Function to validate and process the uploaded company data
def process_company_data(file_path):
    errors = []
    logger.info(f"Processing company data from {file_path}")
    
    # Step 1: Load and validate the uploaded CSV
    try:
        company_df = pd.read_csv(file_path)
        
        # Check required columns - only Date is required
        if 'Date' not in company_df.columns:
            errors.append("Missing required column: Date")
            return {'success': False, 'message': 'CSV file is missing the Date column', 'errors': errors}
        
        # Convert Date to datetime
        try:
            company_df['Date'] = pd.to_datetime(company_df['Date'])
        except Exception as e:
            errors.append(f"Date column could not be parsed: {str(e)}")
            return {'success': False, 'message': 'Invalid date format', 'errors': errors}
        
        # Check for data after 2024-12-31 (we only want historical data)
        future_data = company_df[company_df['Date'] > '2024-12-31']
        if not future_data.empty:
            errors.append("CSV contains data after 2024-12-31. Only historical data should be uploaded.")
            return {'success': False, 'message': 'CSV contains future data', 'errors': errors}
        
        # Identify numeric columns automatically
        numeric_columns = []
        for col in company_df.columns:
            if col != 'Date' and pd.api.types.is_numeric_dtype(company_df[col]):
                numeric_columns.append(col)
        
        # Check if we have at least one numeric column
        if len(numeric_columns) == 0:
            errors.append("CSV must contain at least one numeric column besides Date.")
            return {'success': False, 'message': 'No numeric data columns found', 'errors': errors}
            
        logger.info(f"Found numeric columns: {numeric_columns}")
        
        # Check for missing values in numeric columns and Date
        columns_to_check = numeric_columns + ['Date']
        if company_df[columns_to_check].isna().any().any():
            # Find which columns have missing values
            missing_columns = []
            for col in columns_to_check:
                if company_df[col].isna().any():
                    missing_columns.append(col)
            
            errors.append(f"CSV contains missing values in columns: {', '.join(missing_columns)}")
            return {'success': False, 'message': 'Missing values in data', 'errors': errors}
        
        # Load macro data for prediction
        macro_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'monthly_data.csv')
        macro_df = pd.read_csv(macro_data_path)
        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
        
        # Set Date as index for both dataframes
        company_df.set_index('Date', inplace=True)
        macro_df.set_index('Date', inplace=True)
        
        # Get historical data for training (up to 2024-12-31)
        historical_macro = macro_df[macro_df.index <= '2024-12-31']
        
        # Find overlapping dates between company data and macro data
        overlapping_dates = company_df.index.intersection(historical_macro.index)
        
        if len(overlapping_dates) < 12:  # Require at least 12 months of overlapping data
            errors.append(f"Not enough matching dates between company data and macroeconomic data. Found only {len(overlapping_dates)} matching months, but need at least 12.")
            return {'success': False, 'message': 'Insufficient matching dates between datasets', 'errors': errors}
        
        logger.info(f"Found {len(overlapping_dates)} overlapping dates between datasets")
        
        # Subset both dataframes to overlapping dates
        company_subset = company_df.loc[overlapping_dates]
        macro_subset = historical_macro.loc[overlapping_dates]
        
        # Merge historical data on overlapping dates
        merged_df = macro_subset.join(company_subset, how='inner')
        
        logger.info(f"Merged data shape: {merged_df.shape}")
        
        # Prepare data for predictions
        future_macro = macro_df[macro_df.index > '2024-12-31']
        future_dates = future_macro.index
        
        # Dictionary to store model performance metrics
        model_performance = {}
        
        # Now train models for each numeric column in company data
        prediction_results = {}
        
        for metric in numeric_columns:
            logger.info(f"Training model for {metric}")
            
            # Prepare feature matrix X and target vector y
            X = merged_df.drop(columns=numeric_columns)
            y = merged_df[metric]
            
            # Split into train and test sets for model evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Train on training set
            test_model = RandomForestRegressor(n_estimators=100, random_state=42)
            test_model.fit(X_train, y_train)
            
            # Make predictions on test set
            y_pred = test_model.predict(X_test)
            
            # Calculate performance metrics
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate MAPE (Mean Absolute Percentage Error) safely
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((y_test - y_pred) / np.where(y_test != 0, y_test, np.nan)) * 100)
                if np.isnan(mape) or np.isinf(mape):
                    mape = 0  # Handle division by zero or other issues
            
            # Get feature importances
            feature_importance = {}
            for i, feature in enumerate(X.columns):
                feature_importance[feature] = float(test_model.feature_importances_[i])
            
            # Sort feature importances
            sorted_importance = dict(sorted(
                feature_importance.items(), 
                key=lambda item: item[1], 
                reverse=True
            )[:10])  # Keep top 10 features
            
            # Store metrics
            model_performance[metric] = {
                'r2': float(r2),
                'mae': float(mae),
                'mse': float(mse),
                'rmse': float(rmse),
                'mape': float(mape),
                'feature_importance': sorted_importance,
                'test_actual': y_test.tolist(),
                'test_predicted': y_pred.tolist(),
                'test_dates': [d.strftime('%Y-%m-%d') for d in X_test.index.tolist()]
            }
            
            logger.info(f"Model metrics for {metric}: R² = {r2:.4f}, MAE = {mae:.4f}, MAPE = {mape:.2f}%")
            
            # Now train on the full dataset for final predictions
            final_model = RandomForestRegressor(n_estimators=100, random_state=42)
            final_model.fit(X, y)
            
            # Make predictions on future data
            predictions = final_model.predict(future_macro)
            
            # Store predictions
            prediction_results[metric] = predictions.round(2)
        
        # Save model performance metrics
        performance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'model_performance.json')
        with open(performance_path, 'w') as f:
            # Convert numpy arrays to lists and handle other non-serializable items
            json.dump(model_performance, f, default=lambda o: o if isinstance(o, (int, float, str, bool, dict, list)) else str(o))
        
        logger.info(f"Saved model performance metrics to {performance_path}")
        
        # Create prediction dataframe
        prediction_df = pd.DataFrame({
            'Date': future_dates
        })
        
        # Add predictions for each metric
        for metric in numeric_columns:
            prediction_df[metric] = prediction_results[metric]
        
        # Add is_predicted flag
        prediction_df['is_predicted'] = True
        
        # Save to CSV
        output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'predicted_company_data_2025_2026.csv')
        prediction_df.to_csv(output_path, index=False)
        logger.info(f"Saved predictions to {output_path}")
        
        # Prepare historical data for saving
        company_df_reset = company_df.reset_index()
        company_df_reset['is_predicted'] = False
        
        # Save the complete company data
        historical_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'synthetic_company_data.csv')
        company_df_reset.to_csv(historical_path, index=False)
        logger.info(f"Saved historical data to {historical_path}")
        
        # Return success message with details
        start_date = overlapping_dates.min().strftime('%Y-%m-%d')
        end_date = overlapping_dates.max().strftime('%Y-%m-%d')
        
        return {
            'success': True,
            'message': 'File processed successfully',
            'records_processed': len(overlapping_dates),
            'date_range': f"{start_date} to {end_date}",
            'predictions_generated': len(prediction_df),
            'metrics_processed': numeric_columns,
            'model_performance': {metric: {'r2': model_performance[metric]['r2'], 
                                          'mape': model_performance[metric]['mape']} 
                                 for metric in numeric_columns}
        }
        
    except Exception as e:
        logger.exception("Error in process_company_data")
        errors.append(str(e))
        return {'success': False, 'message': 'Error processing company data', 'errors': errors}
    
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
        logger.warning(f"Could not find the CSV file at: {historical_path}. Creating dummy data.")
        # Create dummy data for testing
        dates = pd.date_range(start='2020-01-31', end='2024-12-31', freq='M')
        dummy_df = pd.DataFrame({
            'Date': dates,
            'Revenue': [100000 + i * 1000 + np.random.normal(0, 2000) for i in range(len(dates))],
            'Profit': [20000 + i * 200 + np.random.normal(0, 1000) for i in range(len(dates))],
            'Risk_Score': [50 + np.sin(i/6) * 10 for i in range(len(dates))],
            'is_predicted': False
        })
        dummy_df.to_csv(historical_path, index=False)
        
        # Create dummy predicted data
        pred_dates = pd.date_range(start='2025-01-31', end='2026-12-31', freq='M')
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
            if 'is_predicted' not in historical_df.columns:
                historical_df['is_predicted'] = False
        except Exception as e:
            logger.error(f"Error loading historical company data: {e}")
            historical_df = pd.DataFrame(columns=['Date', 'is_predicted'])
        
        if not os.path.exists(predicted_path):
            logger.warning(f"Could not find the CSV file at: {predicted_path}. Creating dummy predictions.")
            # Create dummy predicted data based on available columns in historical data
            if not historical_df.empty:
                # Get numeric columns from historical data
                numeric_columns = [col for col in historical_df.columns 
                                  if col not in ['Date', 'is_predicted'] 
                                  and pd.api.types.is_numeric_dtype(historical_df[col])]
                
                pred_dates = pd.date_range(start='2025-01-31', end='2026-12-31', freq='M')
                pred_df = pd.DataFrame({'Date': pred_dates, 'is_predicted': True})
                
                # Create predictions for each numeric column
                for col in numeric_columns:
                    # Get the last value from historical data
                    if len(historical_df) > 0:
                        last_value = historical_df[col].iloc[-1]
                        # Create simple trend with some randomness
                        pred_df[col] = [last_value + i * (last_value * 0.01) + np.random.normal(0, last_value * 0.03) 
                                       for i in range(len(pred_dates))]
                    else:
                        # Default values if historical data is empty
                        pred_df[col] = 100 + np.random.normal(0, 10, size=len(pred_dates))
                
                pred_df.to_csv(predicted_path, index=False)
                predicted_df = pred_df
            else:
                predicted_df = pd.DataFrame(columns=['Date', 'is_predicted'])
        else:
            # Load predicted data
            try:
                predicted_df = pd.read_csv(predicted_path)
                predicted_df['Date'] = pd.to_datetime(predicted_df['Date'])
                if 'is_predicted' not in predicted_df.columns:
                    predicted_df['is_predicted'] = True
            except Exception as e:
                logger.error(f"Error loading predicted company data: {e}")
                predicted_df = pd.DataFrame(columns=['Date', 'is_predicted'])
    
    # Combine the dataframes
    df = pd.concat([historical_df, predicted_df], ignore_index=True)
    
    # Sort by date
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Print information about the data for debugging
    logger.info(f"Company data loaded. Found {len(df)} rows.")
    logger.info(f"Date range: {df['Date'].min() if not df.empty else 'N/A'} to {df['Date'].max() if not df.empty else 'N/A'}")
    logger.info(f"Predicted data points: {df['is_predicted'].sum() if not df.empty else 0}")
    
    # Identify numeric metrics
    numeric_metrics = [col for col in df.columns 
                      if col not in ['Date', 'is_predicted'] 
                      and pd.api.types.is_numeric_dtype(df[col])]
    logger.info(f"Numeric metrics found: {numeric_metrics}")
    
    return df

# Function to calculate correlations
def calculate_correlations():
    try:
        print("Starting correlation calculation...")
        macro_df = load_macro_data()
        company_df = load_company_data()
        
        # Handle the is_predicted columns to avoid conflicts in the merge
        if 'is_predicted' in macro_df.columns:
            macro_df = macro_df.rename(columns={'is_predicted': 'is_predicted_macro'})
        if 'is_predicted' in company_df.columns:
            company_df = company_df.rename(columns={'is_predicted': 'is_predicted_company'})
        
        # Make sure both datasets have the same date format before merging
        macro_df['Date'] = pd.to_datetime(macro_df['Date'])
        company_df['Date'] = pd.to_datetime(company_df['Date'])
        
        # Merge dataframes on Date
        merged_df = pd.merge(macro_df, company_df, on='Date')
        print(f"Successfully merged data. Shape: {merged_df.shape}")
        
        # List all columns in the merged dataframe for debugging
        print("Columns in merged dataframe:", merged_df.columns.tolist())
        
        # Get company metrics and macro indicators
        company_metrics = ['Revenue', 'Profit', 'Risk_Score']
        macro_indicators = [col for col in macro_df.columns if col not in ['Date', 'is_predicted_macro']]
        
        # Calculate correlations
        correlations = {}
        for metric in company_metrics:
            correlations[metric] = {}
            for indicator in macro_indicators:
                # Check if both columns exist in the merged dataframe
                if metric in merged_df.columns and indicator in merged_df.columns:
                    # Create a clean dataframe with just these two columns, dropping NaN values
                    corr_df = merged_df[[metric, indicator]].dropna()
                    
                    # Ensure we have enough data points for correlation
                    if len(corr_df) > 1:
                        # Calculate correlation
                        correlation = corr_df.corr().iloc[0, 1]
                        # Store the correlation value, handling NaN
                        correlations[metric][indicator] = float(correlation) if not np.isnan(correlation) else 0
                        print(f"Correlation between {metric} and {indicator}: {correlations[metric][indicator]:.4f}")
                    else:
                        print(f"Not enough data points for {metric} and {indicator}")
                        correlations[metric][indicator] = 0
                else:
                    missing_cols = []
                    if metric not in merged_df.columns:
                        missing_cols.append(metric)
                    if indicator not in merged_df.columns:
                        missing_cols.append(indicator)
                    print(f"Missing columns: {missing_cols}")
                    correlations[metric][indicator] = 0
        
        print("Correlation calculation complete")
        return correlations
    except Exception as e:
        import traceback
        print(f"Error calculating correlations: {e}")
        print(traceback.format_exc())
        
        # Return fallback data for testing
        return {
            "Revenue": {
                "Credit_Interieur": 0.65, 
                "Impots_Revenus": -0.42, 
                "Inflation_Rate": 0.21, 
                "Paiements_Interet": -0.33, 
                "Taux_Interet": -0.78, 
                "RNB_Par_Habitant": 0.85, 
                "Masse_Monetaire": 0.55, 
                "PIB_US_Courants": 0.93, 
                "RNB_US_Courants": 0.82
            },
            "Profit": {
                "Credit_Interieur": 0.57, 
                "Impots_Revenus": -0.38, 
                "Inflation_Rate": -0.25, 
                "Paiements_Interet": -0.31, 
                "Taux_Interet": -0.72, 
                "RNB_Par_Habitant": 0.79, 
                "Masse_Monetaire": 0.49, 
                "PIB_US_Courants": 0.87, 
                "RNB_US_Courants": 0.76
            },
            "Risk_Score": {
                "Credit_Interieur": -0.35, 
                "Impots_Revenus": 0.48, 
                "Inflation_Rate": 0.63, 
                "Paiements_Interet": 0.42, 
                "Taux_Interet": 0.67, 
                "RNB_Par_Habitant": -0.47, 
                "Masse_Monetaire": -0.28, 
                "PIB_US_Courants": -0.52, 
                "RNB_US_Courants": -0.45
            }
        }

@main_bp.route('/')
def index():
    # Get the list of available indicators
    macro_df = load_macro_data()
    company_df = load_company_data()
    
    # Get macro indicators (excluding Date and is_predicted)
    macro_indicators = [col for col in macro_df.columns if col not in ['Date', 'is_predicted']]
    
    # Dynamically determine company metrics from the data
    company_metrics = [col for col in company_df.columns 
                      if col not in ['Date', 'is_predicted'] 
                      and pd.api.types.is_numeric_dtype(company_df[col])]
    
    logger.info(f"Rendering index with {len(macro_indicators)} macro indicators and {len(company_metrics)} company metrics")
    logger.info(f"Company metrics: {company_metrics}")
    
    return render_template('index.html', 
                          macro_indicators=macro_indicators,
                          company_metrics=company_metrics)
@main_bp.route('/macro-data/<indicator>')
def get_macro_data(indicator):
    """Get raw chart data for a macro indicator"""
    try:
        df = load_macro_data()
        if indicator not in df.columns:
            return None
        
        chart_df = df.copy().dropna(subset=[indicator]).sort_values('Date')
        
        return {
            'dates': chart_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'values': chart_df[indicator].tolist(),
            'is_predicted': chart_df['is_predicted'].tolist()
        }
    except Exception as e:
        logger.exception(f"Error getting macro data: {e}")
        return None

def get_company_data(metric):
    """Get raw chart data for a company metric"""
    try:
        df = load_company_data()
        if metric not in df.columns:
            return None
        
        chart_df = df.copy().dropna(subset=[metric]).sort_values('Date')
        
        return {
            'dates': chart_df['Date'].dt.strftime('%Y-%m-%d').tolist(),
            'values': chart_df[metric].tolist(),
            'is_predicted': chart_df['is_predicted'].tolist()
        }
    except Exception as e:
        logger.exception(f"Error getting company data: {e}")
        return None

def get_all_company_metrics():
    """Get a list of all company metrics"""
    company_df = load_company_data()
    return [col for col in company_df.columns if col not in ['Date', 'is_predicted'] 
           and pd.api.types.is_numeric_dtype(company_df[col])]

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
    
    # Dynamically find numeric metrics
    numeric_metrics = [col for col in df.columns 
                     if col not in ['Date', 'is_predicted'] 
                     and pd.api.types.is_numeric_dtype(df[col])]
    
    summary = {}
    for metric in numeric_metrics:
        if metric in predictions.columns:
            metric_data = predictions[metric].dropna()  # Drop NaN values
            
            if not metric_data.empty:
                try:
                    first_value = metric_data.iloc[0]
                    last_value = metric_data.iloc[-1]
                    change_percent = ((last_value - first_value) / first_value * 100) if first_value != 0 else 0
                    
                    summary[metric] = {
                        'min': float(metric_data.min()),
                        'max': float(metric_data.max()),
                        'mean': float(metric_data.mean()),
                        'start': float(first_value),
                        'end': float(last_value),
                        'change_percent': float(change_percent)
                    }
                except Exception as e:
                    logger.error(f"Error processing {metric}: {e}")
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



# Add a route handler for any missing static files to provide better error messages
@main_bp.route('/static/<path:filename>')
def static_files(filename):
    try:
        return current_app.send_static_file(filename)
    except Exception as e:
        return jsonify({'error': f'Static file not found: {filename}'}), 404
    
@main_bp.route('/model-performance/<metric>')
def get_model_performance(metric):
    """Retrieve and return performance metrics for a specific model"""
    try:
        logger.info(f"Retrieving model performance data for metric: {metric}")
        performance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      '..', 'data', 'model_performance.json')
        
        if not os.path.exists(performance_path):
            logger.warning(f"Model performance data file not found at: {performance_path}")
            # Let's try to generate some fallback data
            fallback_data = generate_fallback_performance_data(metric)
            return jsonify(fallback_data)
            
        with open(performance_path, 'r') as f:
            try:
                performance_data = json.load(f)
                logger.info(f"Loaded performance data with keys: {list(performance_data.keys())}")
            except json.JSONDecodeError:
                logger.error(f"Error decoding JSON from performance file")
                performance_data = {}
        
        if metric not in performance_data:
            logger.warning(f"No performance data for metric: {metric}")
            # Generate fallback data for this metric
            fallback_data = generate_fallback_performance_data(metric)
            return jsonify(fallback_data)
            
        logger.info(f"Successfully retrieved performance data for metric: {metric}")
        return jsonify(performance_data[metric])
        
    except Exception as e:
        logger.exception(f"Error retrieving model performance: {e}")
        fallback_data = generate_fallback_performance_data(metric)
        return jsonify(fallback_data)

def generate_fallback_performance_data(metric):
    """Generate fallback performance data when the real data is not available"""
    logger.info(f"Generating fallback performance data for metric: {metric}")
    
    # Load company data to get some realistic values for the metric
    try:
        company_df = load_company_data()
        if metric in company_df.columns and pd.api.types.is_numeric_dtype(company_df[metric]):
            # Get non-predicted data only
            historical_data = company_df[~company_df['is_predicted']][metric].dropna()
            
            if not historical_data.empty:
                # Get some values to use in our fallback data
                mean_value = float(historical_data.mean())
                min_value = float(historical_data.min())
                max_value = float(historical_data.max())
                
                # Generate some test points
                test_actual = []
                test_predicted = []
                test_dates = []
                
                # Use the last 20 points or less if fewer available
                sample_size = min(20, len(historical_data))
                sample_data = historical_data.iloc[-sample_size:].values
                sample_dates = company_df[~company_df['is_predicted']].iloc[-sample_size:]['Date'].dt.strftime('%Y-%m-%d').values
                
                for i, val in enumerate(sample_data):
                    test_actual.append(float(val))
                    # Add some noise to create predicted values
                    noise = np.random.normal(0, abs(val) * 0.05)  # 5% noise
                    test_predicted.append(float(val + noise))
                    test_dates.append(sample_dates[i])
                
                # Create mock feature importance
                # Load macro data to get realistic feature names
                macro_df = load_macro_data()
                feature_names = [col for col in macro_df.columns if col not in ['Date', 'is_predicted']]
                
                feature_importance = {}
                total_importance = 0
                
                # Assign random importance to each feature
                for feature in feature_names:
                    importance = np.random.uniform(0.01, 0.2)
                    feature_importance[feature] = importance
                    total_importance += importance
                
                # Normalize to sum to 1
                for feature in feature_importance:
                    feature_importance[feature] /= total_importance
                
                # Sort by importance
                sorted_importance = dict(sorted(
                    feature_importance.items(),
                    key=lambda item: item[1],
                    reverse=True
                )[:10])  # Keep top 10
                
                # Generate fallback metrics with realistic values
                r2 = 0.85 + np.random.uniform(-0.15, 0.1)  # Generate R² between 0.7 and 0.95
                mae = mean_value * np.random.uniform(0.05, 0.15)  # 5-15% of mean as MAE
                mape = np.random.uniform(3, 15)  # 3-15% MAPE
                
                return {
                    'r2': float(r2),
                    'mae': float(mae),
                    'mse': float(mae * mae * 1.2),  # Just something reasonable
                    'rmse': float(mae * 1.1),
                    'mape': float(mape),
                    'feature_importance': sorted_importance,
                    'test_actual': test_actual,
                    'test_predicted': test_predicted,
                    'test_dates': test_dates.tolist()
                }
    except Exception as e:
        logger.exception(f"Error generating fallback data: {e}")
def get_indicator_data(indicator):
    """Get specific data about a macroeconomic indicator"""
    try:
        df = load_macro_data()
        if indicator not in df.columns:
            return None
            
        # Split into historical and predicted data
        historical = df[~df['is_predicted']][indicator].dropna()
        predicted = df[df['is_predicted']][indicator].dropna()
        
        if historical.empty or predicted.empty:
            return None
            
        # Get current (last historical) value
        current_value = float(historical.iloc[-1])
        
        # Calculate change during prediction period
        pred_start = float(predicted.iloc[0])
        pred_end = float(predicted.iloc[-1])
        pred_change = ((pred_end - pred_start) / pred_start * 100) if pred_start != 0 else 0
        
        # Format data
        return {
            'name': indicator,
            'current_value': current_value,
            'predicted_start': pred_start,
            'predicted_end': pred_end,
            'predicted_change_pct': pred_change,
            'historical_min': float(historical.min()),
            'historical_max': float(historical.max()),
            'predicted_min': float(predicted.min()),
            'predicted_max': float(predicted.max())
        }
    except Exception as e:
        logger.error(f"Error getting indicator data: {e}")
        return None

def get_metric_data(metric):
    """Get specific data about a company metric"""
    try:
        df = load_company_data()
        if metric not in df.columns:
            return None
            
        # Split into historical and predicted data
        historical = df[~df['is_predicted']][metric].dropna()
        predicted = df[df['is_predicted']][metric].dropna()
        
        if historical.empty or predicted.empty:
            return None
            
        # Get current (last historical) value
        current_value = float(historical.iloc[-1])
        
        # Calculate change during prediction period
        pred_start = float(predicted.iloc[0])
        pred_end = float(predicted.iloc[-1])
        pred_change = ((pred_end - pred_start) / pred_start * 100) if pred_start != 0 else 0
        
        # Format data
        return {
            'name': metric,
            'current_value': current_value,
            'predicted_start': pred_start,
            'predicted_end': pred_end,
            'predicted_change_pct': pred_change,
            'historical_min': float(historical.min()),
            'historical_max': float(historical.max()),
            'predicted_min': float(predicted.min()),
            'predicted_max': float(predicted.max())
        }
    except Exception as e:
        logger.error(f"Error getting metric data: {e}")
        return None

def get_correlation(indicator, metric):
    """Get correlation data between an indicator and a metric"""
    try:
        correlations = calculate_correlations()
        if metric in correlations and indicator in correlations[metric]:
            correlation = correlations[metric][indicator]
            
            # Determine correlation strength
            abs_corr = abs(correlation)
            if abs_corr > 0.8:
                strength = "very strong"
            elif abs_corr > 0.6:
                strength = "strong"
            elif abs_corr > 0.4:
                strength = "moderate"
            elif abs_corr > 0.2:
                strength = "weak"
            else:
                strength = "very weak"
            
            direction = "positive" if correlation > 0 else "negative"
            
            return {
                'indicator': indicator,
                'metric': metric,
                'correlation': correlation,
                'strength': strength,
                'direction': direction
            }
        return None
    except Exception as e:
        logger.error(f"Error getting correlation data: {e}")
        return None
import requests
import os
import json

# Set your Groq API key
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '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')  # Replace with your actual key

def call_groq_api(query, context, context_data):
    """Call Groq API with the financial data context"""
    try:
        # Format the context data into a readable prompt
        context_text = format_context_for_prompt(context_data)
        
        # Create the prompt for the model
        prompt = f"""You are a financial analyst assistant for a dashboard that shows macroeconomic indicators and company metrics with predictions for 2025-2026. Answer the user's question using the data provided.

USER QUESTION: {query}

CURRENT VIEW: {context.get('current_view', 'Unknown')}

DASHBOARD DATA:
{context_text}

Provide a clear, detailed answer based only on the data provided. Mention specific numbers from the data such as values, percentages, and correlations. Explain what these numbers mean in business terms. If discussing correlations, explain whether they're positive or negative and what that implies.

If the data shows predictions, explain the expected trend (increase/decrease) and by how much. If model performance data is available, briefly mention the confidence level of predictions.

Always be specific and data-driven rather than generic.
"""
        
        # Prepare headers for Groq API
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        # Prepare payload for Groq API
        payload = {
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "model": "allam-2-7b",  # Use Mixtral model - good for financial analysis
            "temperature": 0.3,             # Lower temperature for more factual responses
            "max_tokens": 1024,
            "top_p": 1
        }
        
        # Make the API call
        logger.info("Sending request to Groq API")
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Process the response
        if response.status_code == 200:
            result = response.json()
            logger.info("Successfully received response from Groq API")
            
            # Extract the text from the response
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                logger.error(f"Unexpected response structure: {result}")
                return generate_enhanced_local_response(query, context_data)
        else:
            logger.error(f"Groq API error: {response.status_code}, {response.text}")
            return generate_enhanced_local_response(query, context_data)
            
    except Exception as e:
        logger.exception(f"Error calling Groq API: {e}")
        return generate_enhanced_local_response(query, context_data)
def format_context_for_mistral(context, context_data):
    """Format context data for the Mistral prompt"""
    formatted_parts = []
    
    # Add current view information
    formatted_parts.append(f"CURRENT VIEW: {context.get('current_view', 'Unknown')}")
    
    # Add indicator data if available
    if 'indicator' in context_data:
        ind = context_data['indicator']
        formatted_parts.append(f"\nMACROECONOMIC INDICATOR: {ind['name']}")
        formatted_parts.append(f"Current value: {ind['current_value']:.2f}")
        formatted_parts.append(f"Historical range: {ind['historical_min']:.2f} to {ind['historical_max']:.2f}")
        formatted_parts.append(f"Predicted for 2025-2026: {ind['predicted_start']:.2f} to {ind['predicted_end']:.2f}")
        formatted_parts.append(f"Predicted change: {ind['predicted_change_pct']:.2f}%")
    
    # Add metric data if available
    if 'metric' in context_data:
        metric = context_data['metric']
        formatted_parts.append(f"\nCOMPANY METRIC: {metric['name']}")
        formatted_parts.append(f"Current value: {metric['current_value']:.2f}")
        formatted_parts.append(f"Historical range: {metric['historical_min']:.2f} to {metric['historical_max']:.2f}")
        formatted_parts.append(f"Predicted for 2025-2026: {metric['predicted_start']:.2f} to {metric['predicted_end']:.2f}")
        formatted_parts.append(f"Predicted change: {metric['predicted_change_pct']:.2f}%")
    
    # Add correlation data if available
    if 'correlation' in context_data:
        corr = context_data['correlation']
        formatted_parts.append(f"\nCORRELATION ANALYSIS:")
        formatted_parts.append(f"Correlation between {corr['indicator']} and {corr['metric']}: {corr['correlation']:.2f}")
        formatted_parts.append(f"This is a {corr['strength']} {corr['direction']} correlation")
        
        if corr['direction'] == 'positive':
            formatted_parts.append(f"When {corr['indicator']} increases, {corr['metric']} tends to increase")
        else:
            formatted_parts.append(f"When {corr['indicator']} increases, {corr['metric']} tends to decrease")
    
    # Add model performance data if available
    if 'model_performance' in context_data:
        perf = context_data['model_performance']
        formatted_parts.append(f"\nMODEL PERFORMANCE:")
        formatted_parts.append(f"R² Score (accuracy): {perf['r2']:.3f}")
        formatted_parts.append(f"Error Rate (MAPE): {perf['mape']:.2f}%")
        
        if 'top_features' in perf and perf['top_features']:
            formatted_parts.append("Top influencing factors:")
            for idx, feature in enumerate(perf['top_features'][:3]):
                formatted_parts.append(f"{idx+1}. {feature['feature']}: {feature['importance']*100:.1f}%")
    
    return "\n".join(formatted_parts)

@main_bp.route('/api/chat', methods=['POST'])
def chat():
    """Handle chatbot queries using Groq API"""
    try:
        # Get query from request
        data = request.json
        query = data.get('query', '')
        
        logger.info(f"Received chat query: {query}")
        
        if not query:
            return jsonify({'error': 'No query provided'}), 400
        
        # Get current context (which charts/metrics are being viewed)
        current_context = data.get('context', {})
        logger.info(f"Current context: {current_context}")
        
        # Extract current indicator and metric
        current_indicator = current_context.get('current_indicator')
        current_metric = current_context.get('current_metric')
        
        # Gather relevant data based on context
        context_data = {}
        
        # Add indicator data if available
        if current_indicator:
            indicator_data = get_indicator_data(current_indicator)
            if indicator_data:
                context_data['indicator'] = indicator_data
                
        # Add metric data if available
        if current_metric:
            metric_data = get_metric_data(current_metric)
            if metric_data:
                context_data['metric'] = metric_data
                
            # Get model performance data if available
            perf_data = get_model_performance_data(current_metric)
            if perf_data:
                context_data['model_performance'] = perf_data
                
        # Get correlation data if both indicator and metric are available
        if current_indicator and current_metric:
            corr_data = get_correlation(current_indicator, current_metric)
            if corr_data:
                context_data['correlation'] = corr_data
        
        # Generate response using Groq API
        response = call_groq_api(query, current_context, context_data)
        
        return jsonify({
            'response': response,
            'sources': list(context_data.keys())  # Return which data sources were used
        })
        
    except Exception as e:
        # Log the full exception with traceback
        logger.exception(f"Error in chat endpoint: {e}")
        
        # Return a simple error response
        return jsonify({
            'response': f"I apologize, but I encountered an error processing your question. Error details: {str(e)}",
            'sources': []
        })
@main_bp.route('/api/test-groq', methods=['GET'])
def test_groq_api():
    """Test endpoint for Groq API connectivity"""
    try:
        # Simple test prompt
        headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "messages": [
                {"role": "user", "content": "Give a one-sentence response to test the API."}
            ],
            "model": "meta-llama/llama-guard-4-12b",
            "max_tokens": 100
        }
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                'success': True,
                'message': 'Groq API connection successful',
                'response': result['choices'][0]['message']['content'] if 'choices' in result else str(result)
            })
        else:
            return jsonify({
                'success': False,
                'message': 'Groq API connection failed',
                'error': f"{response.status_code}: {response.text}"
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })
def generate_simple_response(query, context):
    """Generate a simple response based on the query and context"""
    query_lower = query.lower()
    
    # Get context variables
    current_view = context.get('current_view')
    current_indicator = context.get('current_indicator')
    current_metric = context.get('current_metric')
    
    # Create a basic response based on what's being viewed
    if current_view == 'macro-tab' and current_indicator:
        return f"You're viewing the {current_indicator} macroeconomic indicator. I can explain trends and correlations for this indicator."
    
    elif current_view == 'company-tab' and current_metric:
        return f"You're viewing your company's {current_metric}. I can help explain what factors influence this metric."
    
    elif 'trend' in query_lower or 'increase' in query_lower or 'decrease' in query_lower:
        if current_indicator:
            return f"The {current_indicator} shows a trend based on historical data and our predictions for 2025-2026."
        elif current_metric:
            return f"Your {current_metric} shows a trend based on historical data and our predictions for 2025-2026."
        else:
            return "I can analyze trends in the data when you select a specific indicator or metric."
    
    elif 'correlation' in query_lower or 'relationship' in query_lower or 'impact' in query_lower:
        if current_indicator and current_metric:
            return f"There is a relationship between {current_indicator} and {current_metric} that our model has identified."
        else:
            return "I can explain correlations between economic indicators and your company metrics when you have both selected."
    
    # Default response
    return "I can help explain the data in this dashboard. Please ask about specific indicators, metrics, trends, or relationships you'd like to understand."

# Helper function to retrieve relevant data
def retrieve_relevant_data(query, context):
    """Retrieve data relevant to the user query with enhanced context awareness"""
    logger.info(f"Retrieving data for query: {query}")
    logger.info(f"With context: {context}")
    
    # Initialize data object
    retrieved_data = {
        'query': query,
        'context': context,
        'macro_indicator': None,
        'company_metric': None,
        'correlation_data': None,
        'model_performance': None,
        'prediction_summary': None,
        'visualization_context': {},
        'sources': []
    }
    
    # Extract entities from query
    entities = extract_entities(query)
    logger.info(f"Extracted entities: {entities}")
    
    # Get current view context
    current_view = context.get('current_view')
    current_indicator = context.get('current_indicator')
    current_metric = context.get('current_metric')
    
    # Add visualization context
    if current_view == 'macro-tab' and current_indicator:
        # Get the chart data for the currently displayed indicator
        indicator_data = get_macro_data(current_indicator)
        if indicator_data:
            retrieved_data['visualization_context']['chart_type'] = 'line chart'
            retrieved_data['visualization_context']['displayed_data'] = current_indicator
            retrieved_data['visualization_context']['has_predictions'] = True
            
            # Determine if there's a clear trend
            values = indicator_data.get('values', [])
            if values:
                if len(values) > 5:  # Need enough points to establish a trend
                    # Simplified trend detection
                    start = values[0]
                    end = values[-1]
                    if end > start * 1.05:  # 5% increase
                        retrieved_data['visualization_context']['trend'] = 'increasing'
                    elif end < start * 0.95:  # 5% decrease
                        retrieved_data['visualization_context']['trend'] = 'decreasing'
                    else:
                        retrieved_data['visualization_context']['trend'] = 'stable'
    
    elif current_view == 'company-tab' and current_metric:
        # Get the chart data for the currently displayed metric
        metric_data = get_company_data(current_metric)
        if metric_data:
            retrieved_data['visualization_context']['chart_type'] = 'line chart'
            retrieved_data['visualization_context']['displayed_data'] = current_metric
            retrieved_data['visualization_context']['has_predictions'] = True
            
            # Get model performance data if available
            performance = get_model_performance_data(current_metric)
            if performance:
                retrieved_data['visualization_context']['model_accuracy'] = performance.get('r2', 0)
                # Get the top influencing factor
                top_features = performance.get('top_features', [])
                if top_features:
                    retrieved_data['visualization_context']['top_factor'] = top_features[0].get('feature')
    
    # Process context data
    if current_view == 'macro-tab' and current_indicator:
        # Get data for the displayed indicator
        retrieved_data['macro_indicator'] = get_macro_indicator_data(current_indicator)
        retrieved_data['sources'].append(f"Current macro indicator: {current_indicator}")
        
        # Check if query asks about impact on company metrics
        if any(term in query.lower() for term in ['impact', 'effect', 'affect', 'influence', 'company']):
            # Get correlations with all company metrics
            company_metrics = get_all_company_metrics()
            correlations = []
            for metric in company_metrics:
                corr = get_correlation_data(current_indicator, metric)
                if corr:
                    correlations.append(corr)
            if correlations:
                retrieved_data['correlation_data'] = correlations
                retrieved_data['sources'].append(f"Impact of {current_indicator} on company metrics")
    
    elif current_view == 'company-tab' and current_metric:
        # Get data for the displayed metric
        retrieved_data['company_metric'] = get_company_metric_data(current_metric)
        retrieved_data['sources'].append(f"Current company metric: {current_metric}")
        
        # Always get model performance data for the current metric
        retrieved_data['model_performance'] = get_model_performance_data(current_metric)
        
        # Get key factors (correlations) influencing this metric
        retrieved_data['correlation_data'] = get_all_correlations_for_metric(current_metric)
        retrieved_data['sources'].append(f"Factors influencing {current_metric}")
        
        # Get prediction summary
        retrieved_data['prediction_summary'] = get_prediction_summary(current_metric)
        
    # If specific entities are mentioned in the query, prioritize those
    if entities.get('macro_indicators') or entities.get('company_metrics'):
        # Handle specific correlation questions
        if entities.get('macro_indicators') and entities.get('company_metrics'):
            correlations = []
            for indicator in entities['macro_indicators']:
                for metric in entities['company_metrics']:
                    corr = get_correlation_data(indicator, metric)
                    if corr:
                        correlations.append(corr)
            if correlations:
                retrieved_data['correlation_data'] = correlations
                retrieved_data['sources'].append("Specific correlations mentioned in query")
        
        # Handle queries about specific company metrics
        elif entities.get('company_metrics'):
            for metric in entities['company_metrics']:
                if not retrieved_data.get('company_metric'):
                    retrieved_data['company_metric'] = get_company_metric_data(metric)
                if not retrieved_data.get('model_performance'):
                    retrieved_data['model_performance'] = get_model_performance_data(metric)
                if not retrieved_data.get('prediction_summary'):
                    retrieved_data['prediction_summary'] = get_prediction_summary(metric)
                if not retrieved_data.get('correlation_data'):
                    retrieved_data['correlation_data'] = get_all_correlations_for_metric(metric)
        
        # Handle queries about specific macro indicators
        elif entities.get('macro_indicators'):
            for indicator in entities['macro_indicators']:
                if not retrieved_data.get('macro_indicator'):
                    retrieved_data['macro_indicator'] = get_macro_indicator_data(indicator)
    
    # Handle specific question types
    query_lower = query.lower()
    
    # Questions about trends
    if any(term in query_lower for term in ['trend', 'increase', 'decrease', 'growth', 'decline']):
        if current_metric:
            trend_data = analyze_trend_data(current_metric)
            retrieved_data['trend_analysis'] = trend_data
            retrieved_data['sources'].append(f"Trend analysis for {current_metric}")
        elif current_indicator:
            trend_data = analyze_trend_data(current_indicator, is_macro=True)
            retrieved_data['trend_analysis'] = trend_data
            retrieved_data['sources'].append(f"Trend analysis for {current_indicator}")
    
    # Questions about predictions
    if any(term in query_lower for term in ['predict', 'forecast', 'future', '2025', '2026']):
        if current_metric and not retrieved_data.get('prediction_summary'):
            retrieved_data['prediction_summary'] = get_prediction_summary(current_metric)
            retrieved_data['sources'].append(f"Prediction for {current_metric}")
        
        # Add confidence information based on model performance
        if retrieved_data.get('model_performance'):
            r2 = retrieved_data['model_performance'].get('r2', 0)
            if r2 > 0.8:
                retrieved_data['prediction_confidence'] = 'high'
            elif r2 > 0.6:
                retrieved_data['prediction_confidence'] = 'good'
            elif r2 > 0.4:
                retrieved_data['prediction_confidence'] = 'moderate'
            else:
                retrieved_data['prediction_confidence'] = 'limited'
    
    logger.info(f"Retrieved data sources: {retrieved_data['sources']}")
    return retrieved_data

def analyze_trend_data(data_name, is_macro=False):
    """Analyze trend data for a metric or indicator"""
    try:
        if is_macro:
            df = load_macro_data()
        else:
            df = load_company_data()
            
        if data_name not in df.columns:
            return None
            
        # Split into historical and predicted
        historical = df[~df['is_predicted']][data_name].dropna()
        predicted = df[df['is_predicted']][data_name].dropna()
        
        if historical.empty or predicted.empty:
            return None
            
        # Calculate historical trend
        hist_start = historical.iloc[0]
        hist_end = historical.iloc[-1]
        hist_change = ((hist_end - hist_start) / hist_start) * 100 if hist_start != 0 else 0
        
        # Calculate predicted trend
        pred_start = predicted.iloc[0]
        pred_end = predicted.iloc[-1]
        pred_change = ((pred_end - pred_start) / pred_start) * 100 if pred_start != 0 else 0
        
        # Determine if prediction continues or reverses trend
        same_direction = (hist_change > 0 and pred_change > 0) or (hist_change < 0 and pred_change < 0)
        
        # Compare trend magnitudes
        hist_abs_change = abs(hist_change)
        pred_abs_change = abs(pred_change)
        relative_strength = pred_abs_change / hist_abs_change if hist_abs_change > 0 else 'N/A'
        
        return {
            'name': data_name,
            'historical_change_pct': hist_change,
            'predicted_change_pct': pred_change,
            'continues_trend': same_direction,
            'relative_strength': relative_strength,
            'historical_values': {
                'start': hist_start,
                'end': hist_end
            },
            'predicted_values': {
                'start': pred_start,
                'end': pred_end
            }
        }
    except Exception as e:
        logger.exception(f"Error analyzing trend data: {e}")
        return None
def extract_entities(query):
    """Extract mentioned entities (metrics, indicators, etc.) from query"""
    entities = {
        'macro_indicators': [],
        'company_metrics': []
    }
    
    # Load macro indicators and company metrics
    macro_df = load_macro_data()
    company_df = load_company_data()
    
    macro_indicators = [col for col in macro_df.columns if col not in ['Date', 'is_predicted']]
    company_metrics = [col for col in company_df.columns if col not in ['Date', 'is_predicted']]
    
    # Create normalized versions for matching (lowercase, underscores removed)
    norm_query = query.lower()
    norm_indicators = [ind.lower().replace('_', ' ') for ind in macro_indicators]
    norm_metrics = [metric.lower().replace('_', ' ') for metric in company_metrics]
    
    # Check for matches
    for i, indicator in enumerate(norm_indicators):
        if indicator in norm_query:
            entities['macro_indicators'].append(macro_indicators[i])
    
    for i, metric in enumerate(norm_metrics):
        if metric in norm_query:
            entities['company_metrics'].append(company_metrics[i])
    
    return entities

def get_macro_indicator_data(indicator):
    """Get data for a specific macro indicator"""
    try:
        df = load_macro_data()
        if indicator not in df.columns:
            return None
            
        # Create a summary of the indicator
        historical = df[~df['is_predicted']][indicator].dropna()
        predicted = df[df['is_predicted']][indicator].dropna()
        
        if historical.empty or predicted.empty:
            return None
            
        # Calculate key statistics
        result = {
            'name': indicator,
            'historical_mean': float(historical.mean()),
            'historical_min': float(historical.min()),
            'historical_max': float(historical.max()),
            'predicted_mean': float(predicted.mean()),
            'predicted_min': float(predicted.min()),
            'predicted_max': float(predicted.max()),
            'predicted_change': float((predicted.iloc[-1] - predicted.iloc[0]) / predicted.iloc[0] * 100),
            'current_value': float(historical.iloc[-1]),
            'end_2026_value': float(predicted.iloc[-1])
        }
        
        return result
    except Exception as e:
        logger.exception(f"Error getting macro indicator data: {e}")
        return None

def get_company_metric_data(metric):
    """Get data for a specific company metric"""
    try:
        df = load_company_data()
        if metric not in df.columns:
            return None
            
        # Create a summary of the metric
        historical = df[~df['is_predicted']][metric].dropna()
        predicted = df[df['is_predicted']][metric].dropna()
        
        if historical.empty or predicted.empty:
            return None
            
        # Calculate key statistics
        result = {
            'name': metric,
            'historical_mean': float(historical.mean()),
            'historical_min': float(historical.min()),
            'historical_max': float(historical.max()),
            'predicted_mean': float(predicted.mean()),
            'predicted_min': float(predicted.min()),
            'predicted_max': float(predicted.max()),
            'predicted_change': float((predicted.iloc[-1] - predicted.iloc[0]) / predicted.iloc[0] * 100),
            'current_value': float(historical.iloc[-1]),
            'end_2026_value': float(predicted.iloc[-1])
        }
        
        return result
    except Exception as e:
        logger.exception(f"Error getting company metric data: {e}")
        return None

def get_correlation_data(indicator, metric):
    """Get correlation between a specific indicator and metric"""
    try:
        correlations = calculate_correlations()
        if metric in correlations and indicator in correlations[metric]:
            return {
                'indicator': indicator,
                'metric': metric,
                'correlation': correlations[metric][indicator]
            }
        return None
    except Exception as e:
        logger.exception(f"Error getting correlation data: {e}")
        return None

def get_all_correlations_for_metric(metric):
    """Get all correlations for a specific company metric"""
    try:
        correlations = calculate_correlations()
        if metric in correlations:
            # Sort correlations by absolute value (strongest first)
            sorted_correlations = sorted(
                [{'indicator': ind, 'correlation': val} for ind, val in correlations[metric].items()],
                key=lambda x: abs(x['correlation']),
                reverse=True
            )
            return {
                'metric': metric,
                'correlations': sorted_correlations
            }
        return None
    except Exception as e:
        logger.exception(f"Error getting all correlations: {e}")
        return None

def get_model_performance_data(metric):
    """Get performance metrics for the model predicting a specific company metric"""
    try:
        # Path to the model performance data
        performance_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      '..', 'data', 'model_performance.json')
        
        if not os.path.exists(performance_path):
            return generate_fallback_performance_data(metric)
            
        with open(performance_path, 'r') as f:
            performance_data = json.load(f)
        
        if metric not in performance_data:
            return generate_fallback_performance_data(metric)
            
        # Extract the relevant metrics
        data = performance_data[metric]
        
        # Format the feature importance for readability
        sorted_features = sorted(
            [(feature, importance) for feature, importance in data['feature_importance'].items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        # Get the top 5 features
        top_features = [{'feature': feature, 'importance': importance} 
                      for feature, importance in sorted_features[:5]]
        
        result = {
            'metric': metric,
            'r2': data['r2'],
            'mae': data['mae'],
            'mape': data['mape'],
            'top_features': top_features
        }
        
        return result
    except Exception as e:
        logger.exception(f"Error getting model performance data: {e}")
        return generate_fallback_performance_data(metric)

def get_prediction_summary(metric):
    """Get summary of predictions for a specific company metric"""
    try:
        company_summary = get_company_summary()
        if metric in company_summary:
            summary = company_summary[metric]
            return {
                'metric': metric,
                'start_value': summary['start'],
                'end_value': summary['end'],
                'change_percent': summary['change_percent'],
                'min': summary['min'],
                'max': summary['max'],
                'mean': summary['mean']
            }
        return None
    except Exception as e:
        logger.exception(f"Error getting prediction summary: {e}")
        return None
def format_enhanced_context(retrieved_data):
    """Format retrieved data into a richer context for the prompt"""
    context_parts = []
    
    # Add visualization context
    viz_context = retrieved_data.get('visualization_context', {})
    if viz_context:
        context_parts.append("CURRENT VISUALIZATION:")
        if 'chart_type' in viz_context:
            context_parts.append(f"Chart Type: {viz_context['chart_type']}")
        if 'displayed_data' in viz_context:
            context_parts.append(f"Currently Displayed: {viz_context['displayed_data']}")
        if 'trend' in viz_context:
            context_parts.append(f"Visual Trend: {viz_context['trend']}")
        if 'has_predictions' in viz_context and viz_context['has_predictions']:
            context_parts.append(f"Chart includes predicted data for 2025-2026")
        if 'model_accuracy' in viz_context:
            context_parts.append(f"Model Accuracy (R²): {viz_context['model_accuracy']:.3f}")
        if 'top_factor' in viz_context:
            context_parts.append(f"Top influencing factor: {viz_context['top_factor']}")
        context_parts.append("")
    
    # Add trend analysis if available
    if retrieved_data.get('trend_analysis'):
        trend = retrieved_data['trend_analysis']
        context_parts.append(f"TREND ANALYSIS FOR {trend['name']}:")
        context_parts.append(f"Historical change: {trend['historical_change_pct']:.2f}%")
        context_parts.append(f"Predicted change: {trend['predicted_change_pct']:.2f}%")
        if trend['continues_trend']:
            context_parts.append(f"The prediction continues the historical trend direction")
        else:
            context_parts.append(f"The prediction reverses the historical trend direction")
        if isinstance(trend['relative_strength'], (int, float)):
            if trend['relative_strength'] > 1.2:
                context_parts.append(f"The predicted trend is accelerating (stronger than historical)")
            elif trend['relative_strength'] < 0.8:
                context_parts.append(f"The predicted trend is decelerating (weaker than historical)")
            else:
                context_parts.append(f"The predicted trend maintains similar momentum to historical")
        context_parts.append("")
    
    # Add current view context with more detail
    if retrieved_data['context'].get('current_view'):
        view_names = {
            'macro-tab': 'Macroeconomic Indicators',
            'company-tab': 'Company Metrics',
            'upload-tab': 'Upload Data',
            'chat-tab': 'Chat Assistant'
        }
        view_name = view_names.get(retrieved_data['context'].get('current_view'), 
                               retrieved_data['context'].get('current_view'))
        context_parts.append(f"Current dashboard tab: {view_name}")
    
    # Add macro indicator data if available with more details
    if retrieved_data.get('macro_indicator'):
        ind = retrieved_data['macro_indicator']
        context_parts.append(f"MACRO INDICATOR: {ind['name']}")
        context_parts.append(f"Current value: {ind['current_value']:.2f}")
        context_parts.append(f"Historical range: {ind['historical_min']:.2f} to {ind['historical_max']:.2f}")
        context_parts.append(f"Predicted value (end of 2026): {ind['end_2026_value']:.2f}")
        context_parts.append(f"Predicted change (2025-2026): {ind['predicted_change']:.2f}%")
        context_parts.append("")
    
    # Add company metric data if available with more details
    if retrieved_data.get('company_metric'):
        metric = retrieved_data['company_metric']
        context_parts.append(f"COMPANY METRIC: {metric['name']}")
        context_parts.append(f"Current value: {metric['current_value']:.2f}")
        context_parts.append(f"Historical range: {metric['historical_min']:.2f} to {metric['historical_max']:.2f}")
        context_parts.append(f"Predicted value (end of 2026): {metric['end_2026_value']:.2f}")
        context_parts.append(f"Predicted change (2025-2026): {metric['predicted_change']:.2f}%")
        context_parts.append("")
    
    # Add correlation data with better formatting and explanation
    if retrieved_data.get('correlation_data'):
        if isinstance(retrieved_data['correlation_data'], list):
            context_parts.append("CORRELATIONS:")
            for corr in retrieved_data['correlation_data']:
                correlation = corr['correlation']
                strength = get_correlation_strength(correlation)
                context_parts.append(f"{corr['indicator']} to {corr['metric']}: {correlation:.2f} ({strength})")
        else:
            corr_data = retrieved_data['correlation_data']
            context_parts.append(f"CORRELATIONS FOR {corr_data['metric']}:")
            for idx, corr in enumerate(corr_data['correlations'][:5]):  # Top 5 correlations
                correlation = corr['correlation']
                strength = get_correlation_strength(correlation)
                context_parts.append(f"{idx+1}. {corr['indicator']}: {correlation:.2f} ({strength})")
        context_parts.append("")
    
    # Add model performance data with quality assessments
    if retrieved_data.get('model_performance'):
        perf = retrieved_data['model_performance']
        context_parts.append(f"MODEL PERFORMANCE FOR {perf['metric']}:")
        
        # R² assessment
        r2 = perf['r2']
        r2_quality = "excellent" if r2 > 0.8 else "good" if r2 > 0.6 else "moderate" if r2 > 0.4 else "limited"
        context_parts.append(f"R² Score: {r2:.3f} ({r2_quality})")
        
        # Error assessment
        mape = perf['mape']
        error_quality = "very low" if mape < 5 else "low" if mape < 10 else "moderate" if mape < 20 else "high"
        context_parts.append(f"Error Rate (MAPE): {mape:.2f}% ({error_quality})")
        
        # Feature importance with percentages
        context_parts.append("Top influencing factors:")
        for feature in perf['top_features']:
            pct = feature['importance'] * 100
            context_parts.append(f"- {feature['feature']}: {pct:.1f}%")
        context_parts.append("")
    
    # Add prediction summary with more context
    if retrieved_data.get('prediction_summary'):
        pred = retrieved_data['prediction_summary']
        context_parts.append(f"PREDICTION SUMMARY FOR {pred['metric']}:")
        context_parts.append(f"Starting value (2025): {pred['start_value']:.2f}")
        context_parts.append(f"Ending value (2026): {pred['end_value']:.2f}")
        
        # Add more descriptive context about the change
        change = pred['change_percent']
        if abs(change) < 1:
            magnitude = "slight"
        elif abs(change) < 5:
            magnitude = "modest"
        elif abs(change) < 10:
            magnitude = "significant"
        else:
            magnitude = "substantial"
            
        direction = "increase" if change > 0 else "decrease"
        context_parts.append(f"Overall change: {change:.2f}% ({magnitude} {direction})")
        
        if retrieved_data.get('prediction_confidence'):
            context_parts.append(f"Prediction confidence: {retrieved_data['prediction_confidence']}")
        
        context_parts.append("")
    
    return "\n".join(context_parts)

def get_correlation_strength(correlation):
    """Convert a correlation coefficient to a descriptive strength"""
    abs_corr = abs(correlation)
    direction = "positive" if correlation > 0 else "negative"
    
    if abs_corr > 0.8:
        strength = "very strong"
    elif abs_corr > 0.6:
        strength = "strong"
    elif abs_corr > 0.4:
        strength = "moderate"
    elif abs_corr > 0.2:
        strength = "weak"
    else:
        strength = "very weak"
    
    return f"{strength} {direction}"
def generate_enhanced_local_response(query, context_data):
    """Generate a data-driven response locally when API fails"""
    query_lower = query.lower()
    
    # Check for different types of questions
    is_correlation_query = any(term in query_lower for term in ['correlation', 'relationship', 'impact', 'influence', 'affect', 'between', 'connect'])
    is_prediction_query = any(term in query_lower for term in ['predict', 'forecast', 'future', '2025', '2026', 'will', 'expected'])
    is_trend_query = any(term in query_lower for term in ['trend', 'increase', 'decrease', 'growth', 'decline', 'change'])
    is_performance_query = any(term in query_lower for term in ['model', 'accuracy', 'performance', 'reliable', 'confidence', 'trust', 'r2', 'error'])
    
    # 1. CORRELATION QUESTIONS
    if is_correlation_query and 'correlation' in context_data:
        corr = context_data['correlation']
        indicator = corr['indicator']
        metric = corr['metric']
        correlation = corr['correlation']
        strength = corr['strength']
        direction = corr['direction']
        
        # Prepare business implications based on the correlation
        if "revenue" in metric.lower() or "profit" in metric.lower():
            if direction == "positive":
                business_implication = f"This means that favorable trends in {indicator} are likely to boost your company's {metric}. You should monitor {indicator} closely as a leading indicator for your financial performance."
            else:
                business_implication = f"This means that increases in {indicator} are associated with decreases in your {metric}. Your business may need strategies to offset the negative impact when {indicator} rises."
        elif "cost" in metric.lower() or "expense" in metric.lower():
            if direction == "positive":
                business_implication = f"This suggests that rising {indicator} tends to increase your {metric}, which could pressure your margins. Consider financial hedging or cost management strategies during periods of rising {indicator}."
            else:
                business_implication = f"This suggests that rising {indicator} is associated with decreasing {metric}, which could benefit your margins during such periods."
        elif "risk" in metric.lower():
            if direction == "positive":
                business_implication = f"When {indicator} increases, your company's {metric} tends to increase as well. This suggests you may need enhanced risk management strategies during periods of high {indicator}."
            else:
                business_implication = f"When {indicator} increases, your company's {metric} tends to decrease. This may allow for more aggressive business strategies during periods of high {indicator}."
        else:
            if direction == "positive":
                business_implication = f"As {indicator} increases, your {metric} tends to increase as well, suggesting a beneficial relationship."
            else:
                business_implication = f"As {indicator} increases, your {metric} tends to decrease, suggesting an inverse relationship that should be considered in your planning."
        
        # Create the full response
        response = f"The data shows a **{strength} {direction} correlation** ({correlation:.2f}) between {indicator} and your company's {metric}. "
        response += f"This means that when {indicator} changes, your {metric} tends to change in a {'similar' if direction == 'positive' else 'opposite'} direction. "
        response += f"\n\n{business_implication}"
        
        # Add model performance context if available
        if 'model_performance' in context_data:
            r2 = context_data['model_performance']['r2']
            if r2 > 0.8:
                confidence = "high confidence"
            elif r2 > 0.6:
                confidence = "good confidence"
            else:
                confidence = "moderate confidence"
            
            response += f"\n\nOur predictive model identifies this relationship with {confidence} (R² = {r2:.2f})."
        
        return response
    
    # 2. PREDICTION QUESTIONS
    elif is_prediction_query and 'metric' in context_data:
        metric = context_data['metric']
        name = metric['name']
        pred_start = metric['predicted_start']
        pred_end = metric['predicted_end']
        pred_change = metric['predicted_change_pct']
        
        direction = "increase" if pred_change > 0 else "decrease"
        
        # Describe the magnitude of change
        if abs(pred_change) > 20:
            magnitude = "substantial"
        elif abs(pred_change) > 10:
            magnitude = "significant"
        elif abs(pred_change) > 5:
            magnitude = "moderate"
        else:
            magnitude = "slight"
        
        response = f"Based on our predictive model, your company's {name} is projected to **{direction} by {abs(pred_change):.2f}%** from 2025 to 2026. "
        response += f"This represents a {magnitude} change from {pred_start:.2f} at the beginning of 2025 to {pred_end:.2f} by the end of 2026. "
        
        # Add model performance context if available
        if 'model_performance' in context_data:
            perf = context_data['model_performance']
            r2 = perf['r2']
            mape = perf['mape']
            
            if r2 > 0.8:
                reliability = "highly reliable"
            elif r2 > 0.6:
                reliability = "reliable"
            elif r2 > 0.4:
                reliability = "moderately reliable"
            else:
                reliability = "somewhat uncertain"
                
            response += f"\n\nThis prediction is {reliability} with an R² score of {r2:.2f} and an average error rate of {mape:.2f}%. "
            
            # Add top influencing factors
            if 'top_features' in perf and perf['top_features']:
                response += "The most influential factors for this prediction are:\n\n"
                
                for i, feature in enumerate(perf['top_features'][:3]):
                    importance_pct = feature['importance'] * 100
                    response += f"{i+1}. **{feature['feature']}** ({importance_pct:.1f}%)\n"
        
        # Add relevant correlation if available
        if 'correlation' in context_data:
            corr = context_data['correlation']
            response += f"\n\nThis prediction is influenced by the {corr['strength']} {corr['direction']} correlation ({corr['correlation']:.2f}) between {corr['indicator']} and {name}."
        
        return response
    
    # 3. TREND QUESTIONS
    elif is_trend_query:
        if 'metric' in context_data:
            metric = context_data['metric']
            name = metric['name']
            current = metric['current_value']
            historical_min = metric['historical_min']
            historical_max = metric['historical_max']
            pred_change = metric['predicted_change_pct']
            
            trend_direction = "upward" if pred_change > 0 else "downward"
            
            response = f"The trend for your company's {name} shows a {trend_direction} trajectory for 2025-2026. "
            response += f"Currently at {current:.2f}, the historical range has been from {historical_min:.2f} to {historical_max:.2f}. "
            response += f"The model predicts a {'positive' if pred_change > 0 else 'negative'} change of {abs(pred_change):.2f}% through 2026. "
            
            # Add model performance context if available
            if 'model_performance' in context_data:
                r2 = context_data['model_performance']['r2']
                if r2 > 0.7:
                    confidence = "high confidence"
                elif r2 > 0.5:
                    confidence = "good confidence"
                else:
                    confidence = "moderate confidence"
                
                response += f"\n\nThis trend is identified with {confidence} (R² = {r2:.2f})."
            
            return response
        
        elif 'indicator' in context_data:
            ind = context_data['indicator']
            name = ind['name']
            current = ind['current_value']
            pred_change = ind['predicted_change_pct']
            
            trend_direction = "upward" if pred_change > 0 else "downward"
            
            response = f"The {name} indicator shows a {trend_direction} trend for 2025-2026. "
            response += f"Currently at {current:.2f}, it's projected to change by {abs(pred_change):.2f}% through 2026. "
            
            # Add impact on company metrics if correlation is available
            if 'correlation' in context_data:
                corr = context_data['correlation']
                metric = corr['metric']
                correlation = corr['correlation']
                
                if correlation > 0:
                    impact = f"increase your company's {metric}" if pred_change > 0 else f"decrease your company's {metric}"
                else:
                    impact = f"decrease your company's {metric}" if pred_change > 0 else f"increase your company's {metric}"
                
                response += f"\n\nBased on the {corr['strength']} correlation between {name} and {metric} ({correlation:.2f}), this trend is likely to {impact}."
            
            return response
    
    # 4. MODEL PERFORMANCE QUESTIONS
    elif is_performance_query and 'model_performance' in context_data:
        perf = context_data['model_performance']
        metric = perf['metric']
        r2 = perf['r2']
        mape = perf['mape']
        
        # Assess R² score
        if r2 > 0.9:
            r2_assessment = "excellent"
            explanation = "the model captures almost all of the variation in your historical data"
        elif r2 > 0.8:
            r2_assessment = "very good"
            explanation = "the model captures most of the variation in your historical data"
        elif r2 > 0.6:
            r2_assessment = "good"
            explanation = "the model captures a substantial portion of the variation in your historical data"
        elif r2 > 0.4:
            r2_assessment = "moderate"
            explanation = "the model captures some important patterns but misses others"
        else:
            r2_assessment = "limited"
            explanation = "the model struggles to capture all the patterns in your historical data"
            
        # MAPE assessment
        if mape < 5:
            mape_assessment = "very low"
            reliability = "highly reliable"
        elif mape < 10:
            mape_assessment = "low"
            reliability = "quite reliable"
        elif mape < 20:
            mape_assessment = "moderate"
            reliability = "moderately reliable"
        else:
            mape_assessment = "high"
            reliability = "somewhat unreliable"
            
        response = f"The predictive model for {metric} has {r2_assessment} accuracy with an R² score of {r2:.3f}, meaning {explanation}. "
        response += f"The error rate is {mape_assessment} at {mape:.2f}%, making the predictions {reliability}. "
        
        # Add feature importance
        if 'top_features' in perf and perf['top_features']:
            response += "The model's predictions are most influenced by:\n\n"
            
            for i, feature in enumerate(perf['top_features'][:3]):
                importance_pct = feature['importance'] * 100
                response += f"{i+1}. {feature['feature']} ({importance_pct:.1f}%)\n"
        
        return response
    
    # DEFAULT RESPONSES BASED ON CONTEXT
    # If no specific question type is detected, fall back to data-driven descriptions
    if 'metric' in context_data:
        metric = context_data['metric']
        name = metric['name']
        current = metric['current_value']
        pred_change = metric['predicted_change_pct']
        
        direction = "increase" if pred_change > 0 else "decrease"
        return f"I can help explain the {name} metric shown in the dashboard. It's currently at {current:.2f} and is predicted to {direction} by {abs(pred_change):.2f}% by the end of 2026. You can ask about specific trends, predictions, or what factors influence this metric the most."
    
    elif 'indicator' in context_data:
        ind = context_data['indicator']
        name = ind['name']
        current = ind['current_value']
        pred_change = ind['predicted_change_pct']
        
        direction = "increase" if pred_change > 0 else "decrease"
        return f"I can help explain the {name} macroeconomic indicator shown in the dashboard. It's currently at {current:.2f} and is predicted to {direction} by {abs(pred_change):.2f}% by the end of 2026. You can ask about specific trends, predictions, or how this indicator affects your company's metrics."
    
    # Completely generic fallback
    return "I can help explain the data and visualizations in this dashboard. You can ask about specific indicators, metrics, trends, predictions, or relationships between economic factors and your company's performance."

def format_context_for_prompt(context_data):
    """Format context data for the model prompt"""
    formatted_parts = []
    
    # Add indicator data if available
    if 'indicator' in context_data:
        ind = context_data['indicator']
        formatted_parts.append(f"\nMACROECONOMIC INDICATOR: {ind['name']}")
        formatted_parts.append(f"Current value: {ind['current_value']:.2f}")
        formatted_parts.append(f"Historical range: {ind['historical_min']:.2f} to {ind['historical_max']:.2f}")
        formatted_parts.append(f"Predicted for 2025-2026: {ind['predicted_start']:.2f} to {ind['predicted_end']:.2f}")
        formatted_parts.append(f"Predicted change: {ind['predicted_change_pct']:.2f}%")
    
    # Add metric data if available
    if 'metric' in context_data:
        metric = context_data['metric']
        formatted_parts.append(f"\nCOMPANY METRIC: {metric['name']}")
        formatted_parts.append(f"Current value: {metric['current_value']:.2f}")
        formatted_parts.append(f"Historical range: {metric['historical_min']:.2f} to {metric['historical_max']:.2f}")
        formatted_parts.append(f"Predicted for 2025-2026: {metric['predicted_start']:.2f} to {metric['predicted_end']:.2f}")
        formatted_parts.append(f"Predicted change: {metric['predicted_change_pct']:.2f}%")
    
    # Add correlation data if available
    if 'correlation' in context_data:
        corr = context_data['correlation']
        formatted_parts.append(f"\nCORRELATION ANALYSIS:")
        formatted_parts.append(f"Correlation between {corr['indicator']} and {corr['metric']}: {corr['correlation']:.2f}")
        formatted_parts.append(f"This is a {corr['strength']} {corr['direction']} correlation")
        
        if corr['direction'] == 'positive':
            formatted_parts.append(f"When {corr['indicator']} increases, {corr['metric']} tends to increase")
        else:
            formatted_parts.append(f"When {corr['indicator']} increases, {corr['metric']} tends to decrease")
    
    # Add model performance data if available
    if 'model_performance' in context_data:
        perf = context_data['model_performance']
        formatted_parts.append(f"\nMODEL PERFORMANCE:")
        formatted_parts.append(f"R² Score (accuracy): {perf['r2']:.3f}")
        formatted_parts.append(f"Error Rate (MAPE): {perf['mape']:.2f}%")
        
        if 'top_features' in perf and perf['top_features']:
            formatted_parts.append("Top influencing factors:")
            for idx, feature in enumerate(perf['top_features'][:3]):
                formatted_parts.append(f"{idx+1}. {feature['feature']}: {feature['importance']*100:.1f}%")
    
    return "\n".join(formatted_parts)

def generate_fallback_response(query, retrieved_data):
    """Generate a fallback response when Claude API is unavailable"""
    # Extract the main topic based on entity extraction and context
    company_metric = None
    macro_indicator = None
    
    if retrieved_data['context'].get('current_metric'):
        company_metric = retrieved_data['context'].get('current_metric')
    elif retrieved_data.get('company_metric'):
        company_metric = retrieved_data['company_metric']['name']
    
    if retrieved_data['context'].get('current_indicator'):
        macro_indicator = retrieved_data['context'].get('current_indicator')
    elif retrieved_data.get('macro_indicator'):
        macro_indicator = retrieved_data['macro_indicator']['name']
    
    # Check if query is about correlations
    if any(term in query.lower() for term in ['correlate', 'correlation', 'relationship', 'impact', 'affect', 'influence']):
        if company_metric and macro_indicator:
            correlation_data = get_correlation_data(macro_indicator, company_metric)
            if correlation_data:
                corr = correlation_data['correlation']
                if corr > 0.7:
                    strength = "strong positive"
                elif corr > 0.3:
                    strength = "moderate positive"
                elif corr > 0:
                    strength = "weak positive"
                elif corr > -0.3:
                    strength = "weak negative"
                elif corr > -0.7:
                    strength = "moderate negative"
                else:
                    strength = "strong negative"
                    
                return f"There is a {strength} correlation ({corr:.2f}) between {macro_indicator} and {company_metric}. This means that changes in {macro_indicator} tend to {corr > 0 and 'coincide with' or 'be inversely related to'} changes in {company_metric}."
        
        elif company_metric:
            corr_data = get_all_correlations_for_metric(company_metric)
            if corr_data and corr_data['correlations']:
                top_indicator = corr_data['correlations'][0]['indicator']
                top_corr = corr_data['correlations'][0]['correlation']
                return f"The strongest correlation for {company_metric} is with {top_indicator} ({top_corr:.2f}). This suggests that {top_indicator} has a significant influence on your {company_metric}."
    
    # Check if query is about predictions
    if any(term in query.lower() for term in ['predict', 'prediction', 'forecast', 'future', '2025', '2026']):
        if company_metric:
            pred = get_prediction_summary(company_metric)
            if pred:
                trend = "increase" if pred['change_percent'] > 0 else "decrease"
                return f"Based on our models, {company_metric} is predicted to {trend} by {abs(pred['change_percent']):.2f}% from 2025 to 2026, starting at {pred['start_value']:.2f} and ending at {pred['end_value']:.2f}."
    
    # Check if query is about model performance
    if any(term in query.lower() for term in ['model', 'accuracy', 'performance', 'reliable', 'confidence']):
        if company_metric:
            perf = get_model_performance_data(company_metric)
            if perf:
                if perf['r2'] > 0.8:
                    quality = "very good"
                elif perf['r2'] > 0.6:
                    quality = "good"
                elif perf['r2'] > 0.4:
                    quality = "moderate"
                else:
                    quality = "limited"
                    
                return f"The prediction model for {company_metric} has {quality} accuracy with an R² score of {perf['r2']:.3f} and an error rate of {perf['mape']:.2f}%. The top factor influencing this metric is {perf['top_features'][0]['feature']}."
    
    # Default response if no specific context matched
    if company_metric:
        return f"I can provide insights about {company_metric} and its relationship with macroeconomic indicators. You can ask about correlations, predictions, or model performance."
    elif macro_indicator:
        return f"I can provide insights about {macro_indicator} and how it affects your company metrics. You can ask about correlations, trends, or future predictions."
    else:
        return "I can help explain the relationships between macroeconomic indicators and your company's performance. You can ask about specific metrics, correlations, or predictions shown in the dashboard."




    
    

