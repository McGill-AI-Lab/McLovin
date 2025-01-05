# forms.py
from django import forms
from .models import UserProfile

#TODO Make the dictionnary dynamically based on the User class
class UserProfileForm(forms.ModelForm):
    class Meta:
        model = UserProfile
        fields = [
            'name',
            'age',
            'grade',
            'ethnicity',
            'faculty',
            'major',
            'bio',
            'preferences',
        ]
        widgets = {
            'ethnicity': forms.CheckboxSelectMultiple(),
            'major': forms.Textarea(attrs={'rows': 2}),
            'bio': forms.Textarea(attrs={'rows': 4}),
            'preferences': forms.Textarea(attrs={'rows': 3}),
        }
