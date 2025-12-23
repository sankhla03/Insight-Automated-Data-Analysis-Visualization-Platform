# dashboard/forms.py
from django import forms

class LoginForm(forms.Form):
    username = forms.CharField(max_length=150)
    password = forms.CharField(widget=forms.PasswordInput)
    confirm_password = forms.CharField(
        widget=forms.PasswordInput, required=False
    )

class UploadForm(forms.Form):
    file = forms.FileField(required=False)
    use_sample = forms.BooleanField(required=False, initial=False)
    do_transform = forms.BooleanField(required=False, initial=False)
    drop_columns = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=False,
        help_text="Select columns to drop"
    )

class VisualizationForm(forms.Form):
    CHART_TYPES = [
        ('histogram', 'Histogram'),
        ('box', 'Box Plot'),
        ('pie', 'Pie Chart'),
        ('line', 'Line Chart'),
        ('map', 'Map Chart (if lat/lng present)')
    ]
    
    chart_type = forms.ChoiceField(choices=CHART_TYPES)
    x_column = forms.ChoiceField(required=False)
    y_column = forms.ChoiceField(required=False)
    color_column = forms.ChoiceField(required=False)

class CorrelationForm(forms.Form):
    method = forms.ChoiceField(
        choices=[
            ('pearson', 'Pearson'),
            ('spearman', 'Spearman'),
            ('kendall', 'Kendall')
        ],
        initial='pearson'
    )
    numeric_columns = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        required=False,
        help_text="Select numeric columns for correlation"
    )

class MLForm(forms.Form):
    target_column = forms.ChoiceField()
    feature_columns = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        help_text="Select feature columns"
    )
    test_size = forms.FloatField(
        initial=0.2,
        min_value=0.1,
        max_value=0.5,
        help_text="Test set size (0.1 to 0.5)"
    )

class AIInsightsForm(forms.Form):
    date_column = forms.ChoiceField(required=False)
    anomaly_threshold = forms.FloatField(
        initial=2.0,
        min_value=1.0,
        max_value=5.0,
        help_text="Z-score threshold for anomaly detection"
    )

class FeedbackForm(forms.Form):
    feedback_text = forms.CharField(
        widget=forms.Textarea,
        max_length=1000,
        help_text="Share your feedback about the application"
    )
