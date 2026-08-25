from django.db import models
from django.contrib.contenttypes.fields import GenericForeignKey
from django.contrib.contenttypes.models import ContentType
from app.models import User
import uuid

class Notification(models.Model):
    TYPE_CHOICES = (
        ('info', 'Info'),
        ('warning', 'Warning'),
        ('success', 'Success'),
        ('error', 'Error'),
    )

    CATEGORY_CHOICES = (
        ('voice_auth_approved', 'Voice Authentication Approved'),
        ('voice_auth_failed', 'Voice Authentication Rejected / Alert'),
        ('voiceprint_enrolled', 'Voiceprint Enrolled / Updated'),
        ('wire_transfer_processed', 'Wire Transfer Processed'),
        ('security_alert', 'Security & Biometrics Alert'),
        ('system_alert', 'System Alert'),
    )

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Who gets it?
    recipient = models.ForeignKey(User, on_delete=models.CASCADE, related_name='app_notifications')
    
    # Who caused it? (Optional, e.g., System updates have no actor)
    actor = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, related_name='actions_caused')
    
    # What is this about? (Optional target object)
    content_type = models.ForeignKey(ContentType, on_delete=models.CASCADE, null=True, blank=True)
    object_id = models.CharField(max_length=255, null=True, blank=True) 
    target = GenericForeignKey('content_type', 'object_id')

    title = models.CharField(max_length=255)
    message = models.TextField()
    
    category = models.CharField(max_length=50, choices=CATEGORY_CHOICES, default='system_alert')
    type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='info')
    
    is_read = models.BooleanField(default=False)
    read_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['recipient', 'is_read']), # Optimize "Unread" queries
        ]

    def __str__(self):
        return f"Notification for {self.recipient.email}: {self.title}"