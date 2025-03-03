import io
import numpy as np

from django.shortcuts import render
from django.http import FileResponse
from pydub import AudioSegment
import pyloudnorm as pyln

from .forms import UploadAudioForm

TARGET_LUFS = -14.0  # Docelowy poziom głośności w LUFS

def musichelper_view(request):
    if request.method == 'POST':
        form = UploadAudioForm(request.POST, request.FILES)
        if form.is_valid():
            # Pobieramy plik z formularza
            audio_file = request.FILES['audio_file']

            # Wczytujemy plik do AudioSegment z użyciem BytesIO
            audio_data = audio_file.read()
            audio_segment = AudioSegment.from_file(io.BytesIO(audio_data))

            # Parametry: częstotliwość próbkowania i liczba kanałów
            sample_rate = audio_segment.frame_rate
            channels = audio_segment.channels

            # Konwersja AudioSegment na tablicę numpy (16-bit)
            samples = np.array(audio_segment.get_array_of_samples())
            
            sample_width = audio_segment.sample_width
            max_val = float(1 << (8 * sample_width - 1))
            samples = samples.astype(np.float32) / max_val
            
            if channels > 1:
                samples = samples.reshape((-1, channels))
            else:
                samples = samples.reshape((-1, 1))

            # Mierzymy obecną głośność (LUFS)
            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(samples)

            # Obliczamy gain dB tak, by osiągnąć TARGET_LUFS
            gain_dB = TARGET_LUFS - loudness

            # Nakładamy gain
            mastered_segment = audio_segment.apply_gain(gain_dB)

            # Przygotowujemy bufor do zapisu przetworzonego pliku
            out_buffer = io.BytesIO()
            # Zapisujemy jako WAV (możesz wybrać MP3, ale pamiętaj o jakości i zależnościach ffmpeg)
            mastered_segment.export(out_buffer, format="wav")
            out_buffer.seek(0)

            # Zwracamy gotowy plik do użytkownika
            return FileResponse(
                out_buffer,
                as_attachment=True,
                filename="mastered_output.wav"
            )
    else:
        form = UploadAudioForm()

    return render(request, 'musichelper/upload.html', {'form': form})
