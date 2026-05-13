# FILE: rag/urls.py

from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('chat/', views.index, name='chat'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),
    path('api/ask/', views.ask, name='ask'),
    path('api/health/', views.health, name='health'),
    path('api/sessions/', views.list_sessions, name='list_sessions'),
    path('api/sessions/create/', views.create_session, name='create_session'),
    path('api/sessions/<str:session_id>/', views.get_session_messages, name='get_session_messages'),
    path('api/sessions/<str:session_id>/delete/', views.delete_session, name='delete_session'),
    path('api/sessions/<str:session_id>/rename/', views.rename_session, name='rename_session'),  # NEW
]