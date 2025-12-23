# dashboard/views.py
from django.shortcuts import render, redirect
from django.contrib import messages
from django.contrib.auth import logout
from django.http import HttpResponse, HttpResponseForbidden
from django.views.decorators.http import require_POST
from .forms import LoginForm, UploadForm, VisualizationForm, CorrelationForm, MLForm, AIInsightsForm, FeedbackForm
from . import utils
import pandas as pd
import plotly.express as px
import json


def login_view(request):
    """
    Login + Register on the same page.
    Uses utils.authenticate_user and utils.register_user.
    """
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            confirm_password = form.cleaned_data.get("confirm_password")

            action = request.POST.get("action")  # "login" or "register"

            if action == "register":
                # Register flow
                if not username or not password:
                    messages.error(request, "Username and password are required.")
                elif password != confirm_password:
                    messages.error(request, "Passwords do not match.")
                else:
                    if utils.register_user(username, password):
                        messages.success(
                            request,
                            "Registration successful! Please login."
                        )
                    else:
                        messages.error(
                            request,
                            "Registration failed. Username may already exist."
                        )
            else:
                # Login flow
                if utils.authenticate_user(username, password):
                    request.session["username"] = username
                    return redirect("dashboard:index")
                else:
                    messages.error(request, "Invalid username or password.")
    else:
        form = LoginForm()

    return render(request, "dashboard/login.html", {"form": form})


def register_view(request):
    """
    Dedicated registration view.
    """
    if request.method == "POST":
        form = LoginForm(request.POST)
        if form.is_valid():
            username = form.cleaned_data["username"]
            password = form.cleaned_data["password"]
            confirm_password = form.cleaned_data.get("confirm_password")

            if not username or not password:
                messages.error(request, "Username and password are required.")
            elif password != confirm_password:
                messages.error(request, "Passwords do not match.")
            else:
                if utils.register_user(username, password):
                    messages.success(
                        request,
                        "Registration successful! Please login."
                    )
                    return redirect("dashboard:login")
                else:
                    messages.error(
                        request,
                        "Registration failed. Username may already exist."
                    )
    else:
        form = LoginForm()

    return render(request, "dashboard/login.html", {"form": form, "show_register": True})


def logout_view(request):
    """
    Simple logout: clear session and go back to login.
    """
    request.session.flush()
    logout(request)
    return redirect("dashboard:login")


def index(request):
    """
    Main dashboard page: comprehensive data analysis interface
    """
    if "username" not in request.session:
        return redirect("dashboard:login")

    context = {"username": request.session["username"]}
    
    # Initialize session data for report sections
    if "report_sections" not in request.session:
        request.session["report_sections"] = []
    if "data_summary" not in request.session:
        request.session["data_summary"] = {}
    if "current_data" not in request.session:
        request.session["current_data"] = {}

    # Handle file upload and data processing
    if request.method == "POST":
        if "upload_data" in request.POST:
            return handle_data_upload(request, context)
        elif "create_visualization" in request.POST:
            return handle_visualization_creation(request, context)
        elif "correlation_analysis" in request.POST:
            return handle_correlation_analysis(request, context)
        elif "ml_analysis" in request.POST:
            return handle_ml_analysis(request, context)
        elif "ai_insights" in request.POST:
            return handle_ai_insights(request, context)
        elif "3d_sculpture" in request.POST:
            return handle_3d_sculpture(request, context)
        elif "funding_analysis" in request.POST:
            return handle_funding_analysis(request, context)
        elif "submit_feedback" in request.POST:
            return handle_feedback_submission(request, context)
    
    # Prepare forms
    context["upload_form"] = UploadForm()
    context["visualization_form"] = VisualizationForm()
    context["correlation_form"] = CorrelationForm()
    context["ml_form"] = MLForm()
    context["ai_insights_form"] = AIInsightsForm()
    context["feedback_form"] = FeedbackForm()
    
    # Load data for dropdowns if available
    current_data = request.session.get("current_data", {})
    if current_data:
        df = pd.read_json(current_data["data"]) if current_data.get("data") else None
        if df is not None and not df.empty:
            context["columns"] = list(df.columns)
            context["numeric_columns"] = list(df.select_dtypes(include=['number']).columns)
            context["date_columns"] = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
    
    return render(request, "dashboard/index.html", context)


def download_cleaned_data(request):
    """Download cleaned data as CSV"""
    if "username" not in request.session:
        return redirect("dashboard:login")
    
    try:
        current_data = request.session.get("current_data", {})
        if not current_data:
            messages.error(request, "No data available to download. Please upload data first.")
            return redirect("dashboard:index")
        
        df = pd.read_json(current_data["data"])
        csv_data = utils.export_to_csv(df)
        
        if csv_data.startswith("Error"):
            messages.error(request, csv_data)
            return redirect("dashboard:index")
        
        response = HttpResponse(csv_data, content_type='text/csv')
        response['Content-Disposition'] = 'attachment; filename="cleaned_data.csv"'
        return response
        
    except Exception as e:
        messages.error(request, f"Error downloading data: {str(e)}")
        return redirect("dashboard:index")

def handle_data_upload(request, context):
    """Handle file upload and data processing"""
    form = UploadForm(request.POST, request.FILES)
    if form.is_valid():
        try:
            use_sample = form.cleaned_data.get("use_sample")
            f = form.cleaned_data.get("file")
            do_transform = form.cleaned_data.get("do_transform")
            drop_columns_list = request.POST.getlist("drop_columns")
            
            # Load data
            if use_sample or not f:
                raw_df = utils.get_sample_data()
            else:
                raw_df = utils.load_data_from_file(f)

            # Clean data
            cleaned_df, summary = utils.clean_data_enhanced(raw_df)
            
            # Drop columns if specified
            if drop_columns_list:
                cleaned_df = utils.drop_columns(cleaned_df, drop_columns_list)
                summary["columns_dropped"] = drop_columns_list
            
            # Transform categorical to numeric if requested
            if do_transform:
                transformed_df, mapping = utils.transform_categorical_to_numeric(cleaned_df)
            else:
                transformed_df, mapping = cleaned_df, {}
            
            # Store data in session
            request.session["current_data"] = {
                "data": transformed_df.to_json(),
                "columns": list(transformed_df.columns),
                "mapping": mapping,
                "summary": summary
            }
            
            # Store data summary for report
            request.session["data_summary"] = summary
            
            # Add to report sections
            report_section = f"""
            <h3>Data Processing Summary</h3>
            <ul>
                <li>Original Rows: {summary['original_rows']}</li>
                <li>Original Columns: {summary['original_columns']}</li>
                <li>Cleaned Rows: {summary['cleaned_rows']}</li>
                <li>Duplicates Removed: {summary['duplicates_removed']}</li>
                <li>Missing Values Filled: {summary['missing_values_filled']}</li>
            </ul>
            """
            report_sections = request.session.get("report_sections", [])
            report_sections.append(report_section)
            request.session["report_sections"] = report_sections
            
            # Update context
            context.update({
                "upload_form": form,
                "data_summary": summary,
                "columns": list(transformed_df.columns),
                "numeric_columns": list(transformed_df.select_dtypes(include=['number']).columns),
                "data_preview": transformed_df.to_html(classes="table table-striped", escape=False),
                "mapping": mapping,
                "total_rows": len(transformed_df)
            })
            
            messages.success(request, "Data processed successfully!")
            
        except Exception as e:
            messages.error(request, f"Error processing data: {str(e)}")
    
    return render(request, "dashboard/index.html", context)


def handle_visualization_creation(request, context):
    """Handle visualization creation"""

    # Prepare form with dynamic column choices
    current_data = request.session.get("current_data", {})
    columns = []

    if current_data and current_data.get("data"):
        try:
            df = pd.read_json(current_data["data"])
            columns = list(df.columns)
        except:
            columns = []

    # Create form with dynamic choices
    VisualizationFormClass = VisualizationForm
    VisualizationFormClass.base_fields['x_column'].choices = [('', 'Auto')] + [(col, col) for col in columns]
    VisualizationFormClass.base_fields['y_column'].choices = [('', 'Auto')] + [(col, col) for col in columns]
    VisualizationFormClass.base_fields['color_column'].choices = [('', 'None')] + [(col, col) for col in columns]

    form = VisualizationFormClass(request.POST)

    if form.is_valid():
        try:
            current_data = request.session.get("current_data", {})
            if not current_data:
                messages.error(request, "Please upload data first")
                return render(request, "dashboard/index.html", context)

            df = pd.read_json(current_data["data"])

            chart_type = form.cleaned_data["chart_type"]
            x_col = form.cleaned_data.get("x_column")
            y_col = form.cleaned_data.get("y_column")
            color_col = form.cleaned_data.get("color_column")

            # ===================== VALIDATION (FIXES BLANK GRAPH) =====================

            if chart_type == "box":
                if not y_col:
                    messages.error(request, "Box plot requires a Y-axis column")
                    return render(request, "dashboard/index.html", context)

                if not pd.api.types.is_numeric_dtype(df[y_col]):
                    messages.error(request, "Y-axis must be numeric for Box Plot")
                    return render(request, "dashboard/index.html", context)

                if x_col == y_col:
                    messages.error(request, "X-axis and Y-axis cannot be the same for Box Plot")
                    return render(request, "dashboard/index.html", context)

            if chart_type == "histogram":
                if x_col and not pd.api.types.is_numeric_dtype(df[x_col]):
                    messages.error(request, "Histogram requires a numeric X-axis")
                    return render(request, "dashboard/index.html", context)

            if chart_type == "scatter":
                if not x_col or not y_col:
                    messages.error(request, "Scatter plot requires both X and Y axes")
                    return render(request, "dashboard/index.html", context)

                if not pd.api.types.is_numeric_dtype(df[x_col]) or not pd.api.types.is_numeric_dtype(df[y_col]):
                    messages.error(request, "Scatter plot requires numeric X and Y axes")
                    return render(request, "dashboard/index.html", context)

            # ===================== VALIDATION ENDS =====================

            fig = utils.create_visualization(df, chart_type, x_col, y_col, color_col)

            if fig:
                chart_html = utils.fig_to_html_components(fig)
                context["chart_html"] = chart_html
                context["visualization_form"] = form

                # Add to report sections
                report_section = f"""
                    <h3>Visualization: {chart_type.title()}</h3>
                    <p>Chart Type: {chart_type}</p>
                    <p>X-axis: {x_col or 'Auto'}</p>
                    <p>Y-axis: {y_col or 'Auto'}</p>
                    {chart_html['div']}
                """

                report_sections = request.session.get("report_sections", [])
                report_sections.append(report_section)
                request.session["report_sections"] = report_sections

                messages.success(request, f"{chart_type.title()} chart created successfully!")
            else:
                messages.error(request, "Failed to create visualization")

        except Exception as e:
            messages.error(request, f"Error creating visualization: {str(e)}")

    return render(request, "dashboard/index.html", context)

def handle_correlation_analysis(request, context):
    """Handle correlation analysis"""
    form = CorrelationForm(request.POST)
    if form.is_valid():
        try:
            current_data = request.session.get("current_data", {})
            if not current_data:
                messages.error(request, "Please upload data first")
                return render(request, "dashboard/index.html", context)
            
            df = pd.read_json(current_data["data"])
            
            method = form.cleaned_data["method"]
            selected_columns = request.POST.getlist("numeric_columns")
            
            image_base64, corr_table = utils.create_correlation_heatmap(df, selected_columns, method)
            
            if image_base64:
                context["correlation_image"] = f"data:image/png;base64,{image_base64}"
                context["correlation_table"] = corr_table
                context["correlation_form"] = form
                
                # Add to report sections
                report_section = f"""
                <h3>Correlation Analysis ({method.title()})</h3>
                <img src="data:image/png;base64,{image_base64}" style="max-width: 100%;">
                <h4>Correlation Matrix</h4>
                {corr_table}
                """
                
                report_sections = request.session.get("report_sections", [])
                report_sections.append(report_section)
                request.session["report_sections"] = report_sections
                
                messages.success(request, "Correlation analysis completed!")
            else:
                messages.error(request, "Failed to create correlation heatmap")
                
        except Exception as e:
            messages.error(request, f"Error in correlation analysis: {str(e)}")
    
    return render(request, "dashboard/index.html", context)


def handle_ml_analysis(request, context):
    """Handle machine learning analysis"""
    form = MLForm(request.POST)
    if form.is_valid():
        try:
            current_data = request.session.get("current_data", {})
            if not current_data:
                messages.error(request, "Please upload data first")
                return render(request, "dashboard/index.html", context)
            
            df = pd.read_json(current_data["data"])
            
            target_column = form.cleaned_data["target_column"]
            feature_columns = request.POST.getlist("feature_columns")
            test_size = form.cleaned_data["test_size"]
            
            if target_column not in df.columns:
                messages.error(request, "Invalid target column selected")
                return render(request, "dashboard/index.html", context)
            
            if not feature_columns:
                messages.error(request, "Please select at least one feature column")
                return render(request, "dashboard/index.html", context)
            
            results, error = utils.run_machine_learning(df, target_column, feature_columns, test_size)
            
            if error:
                messages.error(request, error)
            else:
                context["ml_results"] = results
                context["ml_form"] = form
                
                # Create report section
                if results['model_type'] == 'regression':
                    report_section = f"""
                    <h3>Machine Learning Analysis (Regression)</h3>
                    <h4>Model Performance</h4>
                    <ul>
                        <li>MSE: {results['mse']:.4f}</li>
                        <li>RMSE: {results['rmse']:.4f}</li>
                        <li>R² Score: {results['r2_score']:.4f}</li>
                    </ul>
                    """
                else:
                    report_section = f"""
                    <h3>Machine Learning Analysis (Classification)</h3>
                    <h4>Model Performance</h4>
                    <ul>
                        <li>Accuracy: {results['accuracy']:.4f}</li>
                    </ul>
                    <img src="data:image/png;base64,{results['confusion_matrix_image']}" style="max-width: 100%;">
                    """
                
                report_sections = request.session.get("report_sections", [])
                report_sections.append(report_section)
                request.session["report_sections"] = report_sections
                
                messages.success(request, "Machine learning analysis completed!")
                
        except Exception as e:
            messages.error(request, f"Error in ML analysis: {str(e)}")
    
    return render(request, "dashboard/index.html", context)


def handle_ai_insights(request, context):
    """Handle AI insights analysis"""
    form = AIInsightsForm(request.POST)
    if form.is_valid():
        try:
            current_data = request.session.get("current_data", {})
            if not current_data:
                messages.error(request, "Please upload data first")
                return render(request, "dashboard/index.html", context)
            
            df = pd.read_json(current_data["data"])
            
            date_column = form.cleaned_data.get("date_column")
            anomaly_threshold = form.cleaned_data["anomaly_threshold"]
            
            insights, error = utils.analyze_ai_insights(df, date_column, anomaly_threshold)
            
            if error:
                messages.error(request, error)
            else:
                context["ai_insights"] = insights
                context["ai_insights_form"] = form
                
                # Add to report sections
                report_section = f"""
                <h3>AI Insights Analysis</h3>
                <h4>Anomaly Detection (Threshold: {anomaly_threshold})</h4>
                """
                
                if 'anomaly_detection' in insights and insights['anomaly_detection']:
                    report_section += "<ul>"
                    for col, data in insights['anomaly_detection'].items():
                        report_section += f"<li>{col}: {data['count']} anomalies ({data['percentage']:.1f}%)</li>"
                    report_section += "</ul>"
                else:
                    report_section += "<p>No anomalies detected.</p>"
                
                if 'trend_analysis' in insights and insights['trend_analysis']:
                    report_section += "<h4>Trend Analysis</h4><ul>"
                    for trend in insights['trend_analysis']:
                        report_section += f"<li>{trend['column']}: {trend['trend']} (correlation: {trend['correlation_with_time']:.3f})</li>"
                    report_section += "</ul>"
                
                report_sections = request.session.get("report_sections", [])
                report_sections.append(report_section)
                request.session["report_sections"] = report_sections
                
                messages.success(request, "AI insights analysis completed!")
                
        except Exception as e:
            messages.error(request, f"Error in AI insights: {str(e)}")
    
    return render(request, "dashboard/index.html", context)


def handle_3d_sculpture(request, context):
    """Handle 3D sculpture creation"""
    try:
        current_data = request.session.get("current_data", {})
        if not current_data:
            messages.error(request, "Please upload data first")
            return render(request, "dashboard/index.html", context)
        
        df = pd.read_json(current_data["data"])
        
        column = request.POST.get("3d_column")
        resolution = int(request.POST.get("3d_resolution", 50))
        
        if column not in df.columns:
            messages.error(request, "Invalid column selected for 3D sculpture")
            return render(request, "dashboard/index.html", context)
        
        fig, stl_base64 = utils.create_3d_sculpture(df, column, resolution)
        
        if fig and stl_base64:
            chart_html = utils.fig_to_html_components(fig)
            context["chart_3d_html"] = chart_html
            context["stl_download"] = f"data:application/sla;base64,{stl_base64}"
            
            # Add to report sections
            report_section = f"""
            <h3>3D Data Sculpture: {column}</h3>
            <p>Resolution: {resolution}x{resolution}</p>
            {chart_html['div']}
            <p><a href="data:application/sla;base64,{stl_base64}" download="data_sculpture.stl">Download STL File</a></p>
            """
            
            report_sections = request.session.get("report_sections", [])
            report_sections.append(report_section)
            request.session["report_sections"] = report_sections
            
            messages.success(request, "3D sculpture created successfully!")
        else:
            messages.error(request, "Failed to create 3D sculpture")
            
    except Exception as e:
        messages.error(request, f"Error creating 3D sculpture: {str(e)}")
    
    return render(request, "dashboard/index.html", context)


def handle_funding_analysis(request, context):
    """Handle funding breakdown analysis"""
    try:
        current_data = request.session.get("current_data", {})
        if not current_data:
            messages.error(request, "Please upload data first")
            return render(request, "dashboard/index.html", context)
        
        df = pd.read_json(current_data["data"])
        
        breakdown, error = utils.analyze_funding_breakdown(df)
        
        if error:
            messages.error(request, error)
        else:
            context["funding_breakdown"] = breakdown
            
            # Add to report sections
            report_section = f"""
            <h3>Funding Breakdown Analysis</h3>
            """
            
            for funding_type, data in breakdown.items():
                report_section += f"""
                <h4>{funding_type}</h4>
                <p>Count: {data['count']}</p>
                {data['summary']}
                """
            
            report_sections = request.session.get("report_sections", [])
            report_sections.append(report_section)
            request.session["report_sections"] = report_sections
            
            messages.success(request, "Funding analysis completed!")
            
    except Exception as e:
        messages.error(request, f"Error in funding analysis: {str(e)}")
    
    return render(request, "dashboard/index.html", context)


def handle_feedback_submission(request, context):
    """Handle feedback submission"""
    form = FeedbackForm(request.POST)
    if form.is_valid():
        feedback_text = form.cleaned_data["feedback_text"]
        # In a real application, you would save this to a database
        messages.success(request, "Thank you for your feedback!")
    else:
        messages.error(request, "Please provide valid feedback")
    
    return render(request, "dashboard/index.html", context)


@require_POST
def drop_columns_ajax(request):
    """AJAX endpoint for dropping columns from cleaned data without page reload"""
    try:
        # Get current data from session
        current_data = request.session.get("current_data", {})
        if not current_data:
            return HttpResponse(
                json.dumps({"error": "No data available. Please upload data first."}),
                content_type="application/json"
            )
        
        df = pd.read_json(current_data["data"])
        
        # Get columns to drop
        columns_to_drop = request.POST.getlist("columns_to_drop")
        
        if not columns_to_drop:
            return HttpResponse(
                json.dumps({"error": "No columns selected to drop."}),
                content_type="application/json"
            )
        
        # Drop columns that exist in the DataFrame
        existing_columns = [col for col in columns_to_drop if col in df.columns]
        if not existing_columns:
            return HttpResponse(
                json.dumps({"error": "No valid columns to drop."}),
                content_type="application/json"
            )
        
        # Drop the columns
        df_updated = df.drop(columns=existing_columns, axis=1)
        
        # Update session data
        current_data["data"] = df_updated.to_json()
        current_data["columns"] = list(df_updated.columns)
        request.session["current_data"] = current_data
        
        # Generate new table HTML
        table_html = df_updated.to_html(classes="table table-striped", escape=False)
        
        # Return success response
        return HttpResponse(
            json.dumps({
                "success": True,
                "table_html": table_html,
                "dropped_columns": existing_columns,
                "remaining_columns": list(df_updated.columns),
                "total_rows": len(df_updated),
                "numeric_columns": list(df_updated.select_dtypes(include=['number']).columns),
                "message": f"Successfully dropped {len(existing_columns)} column(s): {', '.join(existing_columns)}"
            }),
            content_type="application/json"
        )
        
    except Exception as e:
        return HttpResponse(
            json.dumps({"error": f"Error dropping columns: {str(e)}"}),
            content_type="application/json"
        )

@require_POST
def create_visualization_ajax(request):
    """AJAX endpoint for creating visualizations without page reload"""
    try:
        # Get current data from session
        current_data = request.session.get("current_data", {})
        if not current_data:
            return HttpResponse(
                json.dumps({"error": "No data available. Please upload data first."}),
                content_type="application/json"
            )
        
        df = pd.read_json(current_data["data"])
        available_columns = list(df.columns)
        
        # Get form data
        chart_type = request.POST.get("chart_type")
        x_col = request.POST.get("x_column")
        y_col = request.POST.get("y_column")
        color_col = request.POST.get("color_column")
        
        if not chart_type:
            return HttpResponse(
                json.dumps({"error": "Chart type is required."}),
                content_type="application/json"
            )
        
        # Validate and clean column selections
        if x_col and x_col not in available_columns:
            x_col = None  # Use auto-selection
        
        if y_col and y_col not in available_columns:
            y_col = None  # Use auto-selection
        
        if color_col and color_col not in available_columns:
            color_col = None  # Use no color
        
        print(f"Creating visualization: chart_type={chart_type}, x_col={x_col}, y_col={y_col}, color_col={color_col}")
        print(f"Available columns: {available_columns}")
        
        # Create visualization
        fig = utils.create_visualization(df, chart_type, x_col, y_col, color_col)
        
        if fig:
            chart_html = utils.fig_to_html_components(fig)
            
            # Return success response with chart components
            return HttpResponse(
                json.dumps({
                    "success": True,
                    "chart_div": chart_html["div"],
                    "chart_script": chart_html["script"],
                    "chart_type": chart_type,
                    "message": f"{chart_type.title()} chart created successfully!"
                }),
                content_type="application/json"
            )
        else:
            return HttpResponse(
                json.dumps({"error": "Failed to create visualization. Please check your data and parameters."}),
                content_type="application/json"
            )
            
    except Exception as e:
        print(f"Visualization error: {str(e)}")  # Debug log
        return HttpResponse(
            json.dumps({"error": f"Error creating visualization: {str(e)}"}),
            content_type="application/json"
        )

def report_view(request):
    """
    Comprehensive report generation page
    """
    if "username" not in request.session:
        return redirect("dashboard:login")
    
    try:
        session_data = {
            "username": request.session["username"],
            "report_sections": request.session.get("report_sections", []),
            "data_summary": request.session.get("data_summary", {}),
        }
        
        html_report = utils.generate_comprehensive_report(session_data)
        
        if request.GET.get("download"):
            response = HttpResponse(html_report, content_type="text/html")
            response["Content-Disposition"] = f"attachment; filename=data_analysis_report_{request.session['username']}.html"
            return response
        
        context = {
            "username": request.session["username"],
            "report_html": html_report,
            "report_sections": request.session.get("report_sections", []),
            "data_summary": request.session.get("data_summary", {})
        }
        
        return render(request, "dashboard/report.html", context)
        
    except Exception as e:
        messages.error(request, f"Error generating report: {str(e)}")
        return redirect("dashboard:index")
