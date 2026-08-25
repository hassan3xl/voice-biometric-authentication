from django.urls import path
from . import views

app_name = 'notifications'

urlpatterns = [
    path('', views.notification_list, name='list'),
    path('<uuid:notification_id>/', views.notification_detail, name='detail'),
    path('<uuid:notification_id>/mark-as-read/', views.mark_as_read, name='mark_as_read'),
    path('mark-all-as-read/', views.mark_all_as_read, name='mark_all_as_read'),
    path('unread-count/', views.get_unread_count, name='unread_count'),
    path('broadcast-test/', views.broadcast_test_notification, name='broadcast_test'),
]