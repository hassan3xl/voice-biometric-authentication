from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.views.decorators.http import require_http_methods
from django.http import JsonResponse
from .models.notification import Notification


@login_required
def notification_list(request):
    """Display all notifications for the current user."""
    notifications = Notification.objects.filter(recipient=request.user).order_by('-created_at')
    unread_count = notifications.filter(is_read=False).count()
    
    context = {
        'notifications': notifications,
        'unread_count': unread_count,
    }
    return render(request, 'notifications/list.html', context)


@login_required
def notification_detail(request, notification_id):
    """Display notification detail."""
    notification = get_object_or_404(Notification, id=notification_id, recipient=request.user)
    
    if not notification.is_read:
        notification.is_read = True
        notification.save()
    
    context = {
        'notification': notification,
    }
    return render(request, 'notifications/detail.html', context)


@login_required
@require_http_methods(["POST"])
def mark_as_read(request, notification_id):
    """Mark a notification as read."""
    notification = get_object_or_404(Notification, id=notification_id, recipient=request.user)
    notification.is_read = True
    notification.save()
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    messages.success(request, 'Notification marked as read.')
    return redirect('notifications:list')


@login_required
@require_http_methods(["POST"])
def mark_all_as_read(request):
    """Mark all notifications as read for the current user."""
    Notification.objects.filter(recipient=request.user, is_read=False).update(is_read=True)
    
    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        return JsonResponse({'success': True})
    
    messages.success(request, 'All notifications marked as read.')
    return redirect('notifications:list')


@login_required
def get_unread_count(request):
    """Get unread notification count and the last 5 unread notifications (AJAX endpoint)."""
    notifications = Notification.objects.filter(recipient=request.user, is_read=False).order_by('-created_at')[:5]
    data = []
    for n in notifications:
        data.append({
            'id': str(n.id),
            'title': n.title,
            'body': n.message,
            'created_at': n.created_at.isoformat(),
        })
    count = Notification.objects.filter(recipient=request.user, is_read=False).count()
    return JsonResponse({
        'unread_count': count,
        'notifications': data
    })


@login_required
@require_http_methods(["POST"])
def broadcast_test_notification(request):
    """Broadcast a test notification to all users in the system."""
    from apps.users.models import User
    from .notification_services import NotificationService
    
    all_users = User.objects.all()
    title = request.POST.get("title", "System Test Broadcast")
    message = request.POST.get("message", f"Real-time test alert broadcasted by {request.user.email}!")
    
    # We pass actor=None to send the notification to ALL users, including the sender
    NotificationService.send_bulk_notification(
        recipients=all_users,
        actor=None,
        title=title,
        message=message,
        target_obj=None,
        category='system_alert'
    )
    
    return JsonResponse({'success': True, 'message': 'Broadcast sent successfully!'})