from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.shortcuts import redirect, render

from app.forms.profile_forms import DoctorProfileForm, PatientProfileForm
from app.models import DoctorProfile, UserRole


@login_required
def edit_profile(request):
    """Allow users to edit their profile (patient or doctor)."""
    user = request.user
    if user.role == UserRole.PATIENT:
        if request.method == 'POST':
            form = PatientProfileForm(request.POST, request.FILES, instance=user)
            if form.is_valid():
                form.save()
                messages.success(request, 'Profile updated successfully.')
                return redirect('app:patient_dashboard')
        else:
            form = PatientProfileForm(instance=user)
        return render(request, 'app/profile_edit.html', {'form': form})

    if user.role == UserRole.DOCTOR:
        profile = getattr(user, 'doctor_profile', None)
        if profile is None:
            profile = DoctorProfile.objects.create(user=user)
        if request.method == 'POST':
            form = DoctorProfileForm(request.POST, request.FILES, instance=profile)
            if form.is_valid():
                form.save(user=user)
                messages.success(request, 'Profile updated successfully.')
                return redirect('app:doctor_dashboard')
        else:
            initial = {
                'first_name': user.first_name,
                'last_name': user.last_name,
                'email': user.email,
            }
            form = DoctorProfileForm(instance=profile, initial=initial)
        return render(request, 'app/profile_edit.html', {'form': form})
