from django.urls import path
from app import views

app_name = 'app'

urlpatterns = [
    # Authentication dashboard
    path('', views.dashboard_view, name='home'),
    path('dashboard/', views.dashboard_view, name='banking_dashboard'),
    path('voice-vault/', views.voice_vault_view, name='voice_vault'),

    # Performance evaluation & security center
    path('evaluation/', views.evaluation_dashboard_view, name='evaluation_dashboard'),
    path('audit-logs/', views.biometric_audit_logs_view, name='audit_logs'),

    # Authentication workflows
    path('accounts/login/', views.login_view, name='accounts_login'),
    path('login/', views.login_view, name='login'),
    path('register/', views.register_view, name='register'),
    path('logout/', views.logout_view, name='logout'),

    # Voice Biometrics APIs
    path('api/voice/challenge/', views.get_voice_challenge_api, name='api_voice_challenge'),
    path('api/voice/login/', views.voice_login_api, name='api_voice_login'),
    path('api/voice/enroll/', views.voice_enroll_api, name='api_voice_enroll'),
    path('api/voice/verify-step-up/', views.voice_verify_step_up_api, name='api_voice_verify_step_up'),
    path('api/voice/mic-test/', views.voice_mic_test_api, name='api_voice_mic_test'),
    path('api/voice/reset/', views.voice_reset_api, name='api_voice_reset'),
    path('api/voice/test-verify/', views.voice_test_verify_api, name='api_voice_test_verify'),

    # Scientific Evaluation & Benchmark APIs
    path('api/evaluation/simulate-threshold/', views.simulate_threshold_api, name='api_simulate_threshold'),
    path('api/evaluation/run-benchmark/', views.run_benchmark_api, name='api_run_benchmark'),
    path('api/evaluation/export-report/', views.export_evaluation_report_api, name='api_export_evaluation_report'),
]
