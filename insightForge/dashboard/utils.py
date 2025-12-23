# dashboard/utils.py
import pandas as pd
import numpy as np
from django.contrib.auth.hashers import make_password, check_password
from pymongo import MongoClient
import certifi
import plotly.graph_objects as go
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error, confusion_matrix, accuracy_score
from scipy import stats
import trimesh
import io
import base64
import json
from datetime import datetime

def load_users():
    try:
        return pd.read_csv("users.csv")
    except FileNotFoundError:
        return pd.DataFrame(columns=["username", "password_hash"])

# Try MongoDB, but allow failure
try:
    client = MongoClient(
        "mongodb+srv://adityasankhla03_db_user:WljKfQMdnBUmbCPC@cluster0.cfjzbmi.mongodb.net/?appName=Cluster0",
        tls=True,
        tlsCAFile=certifi.where(),
        serverSelectionTimeoutMS=5000,  # 5 seconds
    )
    # Trigger a test ping; this will raise if SSL/connection fails
    client.admin.command("ping")
    db = client["Autodash"]
    user_collection = db["Users"]
    mongodb_available = True
except Exception as e:
    print("MongoDB connection failed, using CSV auth instead:", e)
    mongodb_available = False

    class DummyCollection:
        def find_one(self, query): return None
        def insert_one(self, doc): return None

    user_collection = DummyCollection()

def register_user(username, password):
    if not username or not password:
        return False

    if mongodb_available:
        if user_collection.find_one({"username": username}):
            return False
        hashed_password = make_password(password)
        user_collection.insert_one(
            {"username": username, "password_hash": hashed_password}
        )
        return True
    else:
        users_df = load_users()
        if username in users_df["username"].values:
            return False
        hashed_password = make_password(password)
        new_user = pd.DataFrame(
            [[username, hashed_password]],
            columns=["username", "password_hash"],
        )
        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv("users.csv", index=False)
        return True

def authenticate_user(username, password):
    if mongodb_available:
        user_doc = user_collection.find_one({"username": username})
        if not user_doc:
            return False
        hashed_password = user_doc.get("password_hash")
        return check_password(password, hashed_password)
    else:
        users_df = load_users()
        user_row = users_df[users_df["username"] == username]
        if user_row.empty:
            return False
        hashed_password = user_row.iloc[0]["password_hash"]
        return check_password(password, hashed_password)

def get_sample_data():
    """Return sample dataset for demonstration"""
    import numpy as np
    
    # Create a sample dataset
    np.random.seed(42)
    data = {
        'Category': np.random.choice(['A', 'B', 'C', 'D'], 100),
        'Value': np.random.normal(50, 15, 100),
        'Count': np.random.poisson(10, 100),
        'Revenue': np.random.exponential(1000, 100)
    }
    return pd.DataFrame(data)

def load_data_from_file(file):
    """Load data from uploaded file"""
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith(('.xlsx', '.xls')):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        raise ValueError("Unsupported file format. Please upload CSV, Excel, or JSON files.")

def clean_data(df):
    """Basic data cleaning"""
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    
    # Handle missing values
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ['object', 'string']:
            # For string columns, fill NaN with empty string
            df_cleaned[col] = df_cleaned[col].fillna('')
        else:
            # For numeric columns, fill NaN with median
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    return df_cleaned

def transform_categorical_to_numeric(df):
    """Convert categorical columns to numeric using label encoding"""
    mapping = {}
    df_transformed = df.copy()
    
    for col in df_transformed.columns:
        if df_transformed[col].dtype == 'object':
            # Get unique values and create mapping
            unique_vals = df_transformed[col].unique()
            col_mapping = {val: idx for idx, val in enumerate(unique_vals)}
            mapping[col] = col_mapping
            
            # Apply mapping
            df_transformed[col] = df_transformed[col].map(col_mapping)
    
    return df_transformed, mapping

def fig_to_html_components(fig):
    """Convert plotly figure to HTML components"""
    try:
        # Convert figure to HTML with custom div ID
        import re
        
        # Generate a unique but consistent ID for the chart
        import time
        div_id = f"chart_{int(time.time() * 1000)}"  # Unique ID with timestamp
        
        # Convert figure to full HTML
        full_html = fig.to_html(include_plotlyjs='cdn', div_id=div_id)
        
        # Split HTML into head and body parts
        head_match = re.search(r'<head[^>]*>(.*?)</head>', full_html, re.DOTALL | re.IGNORECASE)
        body_match = re.search(r'<body[^>]*>(.*?)</body>', full_html, re.DOTALL | re.IGNORECASE)
        
        if not head_match or not body_match:
            # Fallback: return the full HTML as a simple div
            return {
                'div': f'<div id="{div_id}" class="plotly-graph-div">{full_html}</div>',
                'script': ''
            }
        
        head_content = head_match.group(1)
        body_content = body_match.group(1)
        
        # Extract CSS from head
        css_content = ""
        if 'style' in head_content:
            style_match = re.search(r'<style[^>]*>(.*?)</style>', head_content, re.DOTALL | re.IGNORECASE)
            if style_match:
                css_content = f'<style>{style_match.group(1)}</style>'
        
        # Extract the chart div
        div_match = re.search(rf'<div[^>]*id="{div_id}"[^>]*>(.*?)</div>', body_content, re.DOTALL | re.IGNORECASE)
        if div_match:
            div_html = f'<div id="{div_id}">{div_match.group(1)}</div>'
        else:
            # Fallback: look for any div with the ID
            div_match = re.search(rf'<div[^>]*id="{div_id}"[^>]*>', body_content, re.IGNORECASE)
            if div_match:
                div_html = f'<div id="{div_id}"></div>'
            else:
                div_html = f'<div id="{div_id}"></div>'
        
        # Extract JavaScript from head and body
        js_scripts = []
        
        # Find all script tags
        script_patterns = [
            r'<script[^>]*src="[^"]*plotly\.js[^"]*"[^>]*></script>',
            r'<script[^>]*src="[^"]*plotly-.*?\.js[^"]*"[^>]*></script>',
            r'<script[^>]*>(.*?)</script>'
        ]
        
        # Extract scripts from head
        for pattern in script_patterns[:2]:  # External script patterns
            for match in re.finditer(pattern, head_content, re.DOTALL | re.IGNORECASE):
                js_scripts.append(match.group(0))
        
        # Extract inline scripts that contain Plotly code
        inline_script_pattern = r'<script[^>]*>(.*?)Plotly\.newPlot\([^)]+\).*?</script>'
        for match in re.finditer(inline_script_pattern, body_content, re.DOTALL | re.IGNORECASE):
            script_content = match.group(0)
            # Only include if it's actually about Plotly
            if 'Plotly' in script_content:
                js_scripts.append(script_content)
        
        # Combine all scripts
        script_html = '\n'.join(js_scripts)
        
        # If no scripts found, create a minimal Plotly script
        if not script_html:
            script_html = f'''
            <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
            <script>
                // Plotly chart will be rendered by the main page
            </script>
            '''
        
        return {
            'div': div_html,
            'script': script_html,
            'css': css_content
        }
        
    except Exception as e:
        print(f"Error in fig_to_html_components: {e}")
        import traceback
        traceback.print_exc()
        return {
            'div': f'<div id="chart_error" style="border: 2px solid red; padding: 20px; margin: 20px 0; background-color: #ffe6e6;">Error creating chart: {str(e)}</div>',
            'script': '',
            'css': ''
        }

def clean_data_enhanced(df):
    """Enhanced data cleaning with detailed metrics"""
    original_rows = len(df)
    original_cols = len(df.columns)
    
    # Remove duplicate rows
    df_cleaned = df.drop_duplicates()
    duplicates_removed = original_rows - len(df_cleaned)
    
    # Handle missing values
    missing_values_filled = 0
    for col in df_cleaned.columns:
        if df_cleaned[col].dtype in ['object', 'string']:
            df_cleaned[col] = df_cleaned[col].fillna('Unknown')
            missing_values_filled += df_cleaned[col].isna().sum()
        else:
            df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
            missing_values_filled += df_cleaned[col].isna().sum()
    
    # Normalize column names
    df_cleaned.columns = [col.strip().replace(' ', '_').replace('-', '_') for col in df_cleaned.columns]
    
    # Data summary metrics - convert numpy types to native Python types
    summary = {
        'original_rows': int(original_rows),
        'original_columns': int(original_cols),
        'cleaned_rows': int(len(df_cleaned)),
        'duplicates_removed': int(duplicates_removed),
        'missing_values_filled': int(missing_values_filled),
        'final_columns': list(df_cleaned.columns)
    }
    
    return df_cleaned, summary

def drop_columns(df, columns_to_drop):
    """Drop specified columns from dataframe"""
    if not columns_to_drop:
        return df
    
    columns_to_drop = [col for col in columns_to_drop if col in df.columns]
    df_reduced = df.drop(columns=columns_to_drop, axis=1)
    return df_reduced

def create_visualization(df, chart_type, x_col=None, y_col=None, color_col=None):
    """Create various types of visualizations with robust column validation"""
    if df.empty or chart_type not in ['histogram', 'box', 'pie', 'line', 'map']:
        return None
    
    try:
        available_columns = list(df.columns)
        print(f"Available columns: {available_columns}")
        print(f"Requesting visualization: {chart_type}, x={x_col}, y={y_col}, color={color_col}")
        
        # Validate and clean column selections
        if x_col and x_col not in available_columns:
            print(f"X column '{x_col}' not found, using auto-selection")
            x_col = None
        
        if y_col and y_col not in available_columns:
            print(f"Y column '{y_col}' not found, using auto-selection")
            y_col = None
        
        if color_col and color_col not in available_columns:
            print(f"Color column '{color_col}' not found, using no color")
            color_col = None
        
        # Auto-select columns if none provided
        if not x_col:
            x_col = available_columns[0] if available_columns else None
            print(f"Auto-selected X column: {x_col}")
        
        if not y_col and chart_type in ['box', 'line'] and len(available_columns) > 1:
            y_col = available_columns[1]
            print(f"Auto-selected Y column: {y_col}")
        
        # Ensure we have valid columns before proceeding
        if not x_col or x_col not in available_columns:
            print("No valid X column available")
            return None
        
        if chart_type in ['box', 'line'] and (not y_col or y_col not in available_columns):
            print("No valid Y column available for this chart type")
            return None
        
        # Create the visualization
        if chart_type == 'histogram':
            fig = px.histogram(df, x=x_col, title=f"Distribution of {x_col}")
            if color_col and color_col in available_columns:
                fig = px.histogram(df, x=x_col, color=color_col, title=f"Distribution of {x_col} by {color_col}")
                
        elif chart_type == 'box':
            fig = px.box(df, x=x_col, y=y_col, title=f"Box Plot: {x_col} vs {y_col}")
            if color_col and color_col in available_columns:
                fig = px.box(df, x=x_col, y=y_col, color=color_col, title=f"Box Plot: {x_col} vs {y_col} by {color_col}")
                
        elif chart_type == 'pie':
            fig = px.pie(df, names=x_col, title=f"Pie Chart: {x_col}")
            if color_col and color_col in available_columns:
                fig = px.pie(df, names=x_col, color=color_col, title=f"Pie Chart: {x_col} by {color_col}")
                
        elif chart_type == 'line':
            fig = px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col} over {x_col}")
            if color_col and color_col in available_columns:
                fig = px.line(df, x=x_col, y=y_col, color=color_col, title=f"Line Chart: {y_col} over {x_col} by {color_col}")
                
        elif chart_type == 'map':
            # Check for latitude/longitude columns
            lat_cols = [col for col in available_columns if 'lat' in col.lower()]
            lng_cols = [col for col in available_columns if 'lng' in col.lower() or 'lon' in col.lower()]
            
            if lat_cols and lng_cols:
                fig = px.scatter_mapbox(df, lat=lat_cols[0], lon=lng_cols[0], hover_name=x_col, title="Geographic Map")
                fig.update_layout(mapbox_style="open-street-map")
                if color_col and color_col in available_columns:
                    fig = px.scatter_mapbox(df, lat=lat_cols[0], lon=lng_cols[0], hover_name=x_col, color=color_col, title="Geographic Map by " + color_col)
                    fig.update_layout(mapbox_style="open-street-map")
            else:
                # Fallback to scatter plot if no lat/lon columns
                fig = px.scatter(df, x=x_col, y=y_col, title=f"Scatter Plot: {x_col} vs {y_col}")
                if color_col and color_col in available_columns:
                    fig = px.scatter(df, x=x_col, y=y_col, color=color_col, title=f"Scatter Plot: {x_col} vs {y_col} by {color_col}")
        
        print("Visualization created successfully")
        return fig
        
    except Exception as e:
        print(f"Error creating visualization: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_correlation_heatmap(df, columns=None, method='pearson'):
    """Create correlation heatmap and return as base64 image"""
    try:
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if columns:
            numeric_cols = [col for col in columns if col in numeric_cols]
        
        if len(numeric_cols) < 2:
            return None, "Need at least 2 numeric columns for correlation analysis"
        
        # Calculate correlation matrix
        corr_matrix = df[numeric_cols].corr(method=method)
        
        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, linewidths=0.5)
        plt.title(f'{method.capitalize()} Correlation Matrix')
        plt.tight_layout()
        
        # Convert to base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        
        return image_base64, corr_matrix.to_html(classes='table table-striped')
    except Exception as e:
        return None, f"Error creating correlation heatmap: {str(e)}"

def create_3d_sculpture(df, column, resolution=50):
    """Create 3D data sculpture and export as STL"""
    try:
        # Get numeric data
        numeric_data = df[column].dropna()
        if len(numeric_data) < 4:
            return None, "Not enough numeric data for 3D sculpture"
        
        # Create grid for 3D surface
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X, Y = np.meshgrid(x, y)
        
        # Normalize data and create surface
        data_normalized = (numeric_data - numeric_data.min()) / (numeric_data.max() - numeric_data.min())
        
        # Create Z values based on data
        Z = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                idx = int((i * resolution + j) % len(data_normalized))
                Z[i, j] = data_normalized.iloc[idx]
        
        # Create 3D surface with plotly
        fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y, colorscale='Viridis')])
        fig.update_layout(
            title=f'3D Data Sculpture: {column}',
            autosize=False,
            width=800,
            height=600,
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title=column,
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )
        
        # Create STL mesh
        vertices = []
        faces = []
        
        for i in range(resolution - 1):
            for j in range(resolution - 1):
                # Create two triangles for each grid cell
                v1 = [X[i, j], Y[i, j], Z[i, j]]
                v2 = [X[i+1, j], Y[i+1, j], Z[i+1, j]]
                v3 = [X[i, j+1], Y[i, j+1], Z[i, j+1]]
                v4 = [X[i+1, j+1], Y[i+1, j+1], Z[i+1, j+1]]
                
                vertices.extend([v1, v2, v3, v4])
                
                # Add faces (triangles)
                base_idx = len(vertices) - 4
                faces.append([base_idx, base_idx + 1, base_idx + 2])
                faces.append([base_idx + 1, base_idx + 3, base_idx + 2])
        
        # Create trimesh object
        vertices_array = np.array(vertices)
        faces_array = np.array(faces)
        
        mesh = trimesh.Trimesh(vertices=vertices_array, faces=faces_array)
        
        # Export as STL
        stl_buffer = io.BytesIO()
        mesh.export(stl_buffer, file_type='stl')
        stl_base64 = base64.b64encode(stl_buffer.getvalue()).decode()
        
        return fig, stl_base64
    except Exception as e:
        return None, f"Error creating 3D sculpture: {str(e)}"

def run_machine_learning(df, target_column, feature_columns, test_size=0.2):
    """Run machine learning analysis"""
    try:
        # Prepare data
        target = df[target_column].dropna()
        features = df[feature_columns].dropna()
        
        # Align indices
        common_idx = target.index.intersection(features.index)
        target = target[common_idx]
        features = features[common_idx]
        
        if len(common_idx) < 10:
            return None, "Not enough data for machine learning analysis"
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=test_size, random_state=42
        )
        
        # Determine if regression or classification
        is_regression = target.dtype in ['int64', 'float64'] and len(target.unique()) > 10
        
        results = {}
        
        if is_regression:
            # Linear Regression
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            
            results['model_type'] = 'regression'
            results['mse'] = mse
            results['rmse'] = rmse
            results['r2_score'] = model.score(X_test, y_test)
            
            # Create actual vs predicted plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=y_test,
                y=y_pred,
                mode='markers',
                name='Predictions',
                text=[f'Actual: {a:.2f}<br>Predicted: {p:.2f}' for a, p in zip(y_test, y_pred)],
                hovertemplate='%{text}<extra></extra>'
            ))
            fig.add_trace(go.Scatter(
                x=[y_test.min(), y_test.max()],
                y=[y_test.min(), y_test.max()],
                mode='lines',
                name='Perfect Prediction',
                line=dict(dash='dash')
            ))
            fig.update_layout(
                title='Actual vs Predicted Values',
                xaxis_title='Actual Values',
                yaxis_title='Predicted Values',
                showlegend=True
            )
            
            results['plot'] = fig
            results['model'] = model
            
        else:
            # Random Forest Classification
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            cm = confusion_matrix(y_test, y_pred)
            
            results['model_type'] = 'classification'
            results['accuracy'] = accuracy
            results['confusion_matrix'] = cm
            
            # Create confusion matrix heatmap
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual')
            plt.xlabel('Predicted')
            
            # Convert to base64
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=150, bbox_inches='tight')
            buffer.seek(0)
            cm_base64 = base64.b64encode(buffer.getvalue()).decode()
            plt.close()
            
            results['confusion_matrix_image'] = cm_base64
            results['model'] = model
        
        return results, None
    except Exception as e:
        return None, f"Error in machine learning analysis: {str(e)}"

def analyze_ai_insights(df, date_column=None, anomaly_threshold=2.0):
    """Generate AI insights including trend analysis and anomaly detection"""
    try:
        insights = {}
        
        # Trend analysis if date column exists
        if date_column and date_column in df.columns:
            # Try to convert to datetime
            try:
                df[date_column] = pd.to_datetime(df[date_column])
                df_sorted = df.sort_values(date_column)
                
                trend_results = []
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                
                for col in numeric_cols:
                    if col != date_column:
                        correlation = df_sorted[col].corr(df_sorted[date_column].astype(int))
                        trend_results.append({
                            'column': col,
                            'correlation_with_time': correlation,
                            'trend': 'increasing' if correlation > 0.1 else 'decreasing' if correlation < -0.1 else 'stable'
                        })
                
                insights['trend_analysis'] = trend_results
                
                # Create time series plot for most correlated column
                if trend_results:
                    most_correlated = max(trend_results, key=lambda x: abs(x['correlation_with_time']))
                    fig = px.line(df_sorted, x=date_column, y=most_correlated['column'], 
                                title=f'Time Series: {most_correlated["column"]} (Correlation: {most_correlated["correlation_with_time"]:.3f})')
                    insights['time_series_plot'] = fig
                
            except Exception as e:
                insights['trend_error'] = f"Could not perform trend analysis: {str(e)}"
        
        # Anomaly detection using z-scores
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        anomalies = {}
        
        for col in numeric_cols:
            data = df[col].dropna()
            z_scores = np.abs(stats.zscore(data))
            anomaly_indices = np.where(z_scores > anomaly_threshold)[0]
            
            if len(anomaly_indices) > 0:
                anomalies[col] = {
                    'count': len(anomaly_indices),
                    'percentage': (len(anomaly_indices) / len(data)) * 100,
                    'threshold': anomaly_threshold,
                    'sample_anomalies': data.iloc[anomaly_indices[:5]].tolist()  # First 5 anomalies
                }
        
        insights['anomaly_detection'] = anomalies
        
        return insights, None
    except Exception as e:
        return None, f"Error in AI insights analysis: {str(e)}"

def analyze_funding_breakdown(df):
    """Analyze funding breakdown if funding_type column exists"""
    try:
        funding_cols = [col for col in df.columns if 'funding' in col.lower() and 'type' in col.lower()]
        
        if not funding_cols:
            return None, "No funding_type column found"
        
        funding_col = funding_cols[0]
        breakdown = {}
        
        # Group by funding type
        for funding_type in df[funding_col].unique():
            if pd.isna(funding_type):
                continue
                
            subset = df[df[funding_col] == funding_type]
            
            # Find common fields for breakdown
            key_fields = ['student_id', 'faculty_id', 'amount', 'date', 'description']
            available_fields = [field for field in key_fields if field in subset.columns]
            
            breakdown[funding_type] = {
                'count': len(subset),
                'fields': available_fields,
                'summary': subset[available_fields].describe().to_html(classes='table table-striped') if available_fields else "No key fields available"
            }
        
        return breakdown, None
    except Exception as e:
        return None, f"Error in funding analysis: {str(e)}"

def generate_comprehensive_report(session_data):
    """Generate comprehensive HTML report"""
    try:
        report_sections = session_data.get('report_sections', [])
        
        html_report = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>InsightForge - Data Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: #f8f9fa; padding: 20px; border-radius: 8px; margin-bottom: 20px; }}
                .section {{ margin-bottom: 30px; padding: 20px; border: 1px solid #ddd; border-radius: 8px; }}
                .chart {{ margin: 20px 0; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>InsightForge - Data Analysis Report</h1>
                <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                <p>Analyst: {session_data.get('username', 'Unknown')}</p>
            </div>
        """
        
        # Add data summary
        if 'data_summary' in session_data:
            summary = session_data['data_summary']
            html_report += f"""
            <div class="section">
                <h2>Data Summary</h2>
                <ul>
                    <li>Original Rows: {summary.get('original_rows', 'N/A')}</li>
                    <li>Original Columns: {summary.get('original_columns', 'N/A')}</li>
                    <li>Cleaned Rows: {summary.get('cleaned_rows', 'N/A')}</li>
                    <li>Duplicates Removed: {summary.get('duplicates_removed', 'N/A')}</li>
                    <li>Missing Values Filled: {summary.get('missing_values_filled', 'N/A')}</li>
                </ul>
            </div>
            """
        
        # Add all report sections
        for section in report_sections:
            html_report += f"""
            <div class="section">
                {section}
            </div>
            """
        
        html_report += """
        </body>
        </html>
        """
        
        return html_report
    except Exception as e:
        return f"Error generating report: {str(e)}"

def export_to_csv(df, filename="cleaned_data.csv"):
    """Export DataFrame to CSV format"""
    try:
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)
        return csv_buffer.getvalue()
    except Exception as e:
        return f"Error exporting to CSV: {str(e)}"
