from django.contrib import admin
from .models import Notification

@admin.register(Notification)
class NotificationAdmin(admin.ModelAdmin):
    list_display = ('id', 'recipient', 'actor', 'title', 'category', 'type', 'is_read', 'created_at')
    list_filter = ('type', 'category', 'is_read', 'created_at')
    search_fields = ('title', 'message', 'recipient__email', 'actor__email')
    ordering = ('-created_at',)
    readonly_fields = ('id', 'created_at')