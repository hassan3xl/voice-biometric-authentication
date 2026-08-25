import threading
import time
import urllib.request
from django.contrib import admin
from django.http import JsonResponse
from django.urls import include, path
from django.utils import timezone


def ping_server(request):
    """Health check view to ping the server and return uptime status."""
    return JsonResponse({
        'status': 'ok',
        'message': 'Server is active and healthy',
        'timestamp': timezone.now().isoformat()
    })


def _ping_loop():
    """Background daemon task that pings the server every 5 minutes (300 seconds)."""
    time.sleep(5)
    while True:
        try:
            req = urllib.request.Request("http://127.0.0.1:8000/ping/", headers={'User-Agent': 'Server-Ping-Bot/1.0'})
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass
        time.sleep(300)  # Ping every 5 minutes


# Ensure background ping worker thread is initialized once
if not getattr(threading, '_server_ping_started', False):
    setattr(threading, '_server_ping_started', True)
    ping_thread = threading.Thread(target=_ping_loop, daemon=True)
    ping_thread.start()


urlpatterns = [
    path('ping/', ping_server, name='ping_server'),
    path('admin/', admin.site.urls),
    path('notifications/', include('notifications.urls')),
    path('', include('app.urls')),
]
