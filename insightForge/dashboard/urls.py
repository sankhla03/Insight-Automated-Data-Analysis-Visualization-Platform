from django.urls import path
from . import views

app_name = "dashboard"

urlpatterns = [
    path("", views.index, name="index"),
    path("login/", views.login_view, name="login"),
    path("register/", views.register_view, name="register"),
    path("logout/", views.logout_view, name="logout"),
    path("report/", views.report_view, name="report"),
    path("download/", views.download_cleaned_data, name="download"),
    path("ajax/visualization/", views.create_visualization_ajax, name="create_visualization_ajax"),
    path("ajax/drop-columns/", views.drop_columns_ajax, name="drop_columns_ajax"),
]
