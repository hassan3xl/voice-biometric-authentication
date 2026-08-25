from decimal import Decimal
from django import forms
from django.contrib.auth import get_user_model

User = get_user_model()


class BankingRegistrationForm(forms.ModelForm):
    first_name = forms.CharField(
        required=True,
        label="First Name",
        widget=forms.TextInput(attrs={
            'placeholder': 'Alexander',
            'class': 'form-input',
        })
    )
    last_name = forms.CharField(
        required=True,
        label="Last Name",
        widget=forms.TextInput(attrs={
            'placeholder': 'Hamilton',
            'class': 'form-input',
        })
    )
    email = forms.EmailField(
        required=True,
        label="Email address",
        widget=forms.EmailInput(attrs={
            'placeholder': 'alexander@example.com',
            'class': 'form-input',
        })
    )
    password1 = forms.CharField(
        label="Password",
        widget=forms.PasswordInput(attrs={
            'placeholder': '••••••••••••',
            'class': 'form-input',
        })
    )
    password2 = forms.CharField(
        label="Confirm password",
        widget=forms.PasswordInput(attrs={
            'placeholder': '••••••••••••',
            'class': 'form-input',
        })
    )

    class Meta:
        model = User
        fields = ['first_name', 'last_name', 'email']

    def clean_email(self):
        email = self.cleaned_data.get('email').strip().lower()
        if User.objects.filter(email__iexact=email).exists() or User.objects.filter(username__iexact=email).exists():
            raise forms.ValidationError("An account with this email address already exists.")
        return email

    def clean(self):
        cleaned_data = super().clean()
        p1 = cleaned_data.get('password1')
        p2 = cleaned_data.get('password2')
        if p1 and p2 and p1 != p2:
            self.add_error('password2', "Passwords do not match.")
        return cleaned_data

    def save(self, commit=True):
        user = super().save(commit=False)
        email = self.cleaned_data['email'].strip().lower()
        user.username = email
        user.email = email
        user.first_name = self.cleaned_data['first_name'].strip()
        user.last_name = self.cleaned_data['last_name'].strip()
        
        user.set_password(self.cleaned_data['password1'])
        if commit:
            user.save()
        return user


class BankingLoginForm(forms.Form):
    username_or_email = forms.CharField(
        required=True,
        label="Email or account number",
        widget=forms.TextInput(attrs={
            'placeholder': 'alexander@example.com or 4012345678',
            'class': 'form-input',
            'id': 'login-identifier',
        })
    )
    password = forms.CharField(
        required=True,
        label="Password",
        widget=forms.PasswordInput(attrs={
            'placeholder': '••••••••••••',
            'class': 'form-input',
            'id': 'login-password',
        })
    )
