from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('home', views.index, name='index'),
    path('app', views.app, name='app'),
    path('add', views.add, name='add'),
    path('results', views.results, name='results'),
    path('results_good', views.results, name='results_good'),
]