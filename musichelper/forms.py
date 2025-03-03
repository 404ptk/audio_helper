from django import forms

class UploadAudioForm(forms.Form):
    audio_file = forms.FileField(
        required=True,
        label='Wybierz plik audio (MP3/WAV)'
    )