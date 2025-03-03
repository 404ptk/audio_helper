import io
import numpy as np
import librosa

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

def analyze_track_view(request):
    """
    Widok, który przyjmuje plik audio i zwraca
    wykryte tempo (BPM) oraz przybliżoną tonację (np. C minor).
    """
    key_info = None
    bpm_info = None

    if request.method == 'POST':
        form = UploadAudioForm(request.POST, request.FILES)
        if form.is_valid():
            audio_file = request.FILES['audio_file']
            audio_data = audio_file.read()

            # Wczytujemy audio za pomocą librosy
            y, sr = librosa.load(io.BytesIO(audio_data), sr=None, mono=True)

            # Wykrywanie tempa (zwraca tablicę, najczęściej 1 element)
            tempo_array, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            tempo_value = tempo_array[0]  # wyciągamy pierwszy element
            bpm_info = round(tempo_value)

            # Wykrywanie tonacji (key) - proste podejście
            # 1) Obliczamy chromę (12-elementowy wektor na klatkę)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            # 2) Uśredniamy po czasie
            chroma_mean = np.mean(chroma, axis=1)
            # 3) Szacowanie tonacji: porównanie z wzorcami major/minor (dość przybliżone)
            key_info = estimate_key(chroma_mean)

    else:
        form = UploadAudioForm()

    return render(request, 'musichelper/analyze.html', {
        'form': form,
        'key_info': key_info,
        'bpm_info': bpm_info,
    })


# ----------------------------------------------------
# Proste, przybliżone rozpoznawanie tonacji na podstawie chromy
# ----------------------------------------------------
# Dla uproszczenia zdefiniujemy "wzorce" major/minor
# w postaci 12-elementowego wektora, który reprezentuje typowe
# składowe w danej tonacji. Potem przesuniemy je o 12 możliwych
# nut (C, C#, D, ..., B) i znajdziemy największą korelację.
MAJOR_PROFILE = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0,
                        1.0, 0.0, 0.0, 0.0])  # uproszczony
MINOR_PROFILE = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0,
                        0.0, 0.0, 0.0, 0.0])  # uproszczony

NOTES = ["C", "C#", "D", "D#", "E", "F",
        "F#", "G", "G#", "A", "A#", "B"]

def estimate_key(chroma_mean):
    """
    Bardzo uproszczony algorytm: 
    - Bierzemy znormalizowaną średnią chromę
    - Porównujemy z 12 przesunięciami (rotacjami) wzorca major i minor
    - Wybieramy najwyższą korelację
    """
    # Upewniamy się, że nasz wektor nie jest zerowy
    if np.sum(chroma_mean) == 0:
        return "Unknown"

    # Normalizacja
    chroma_norm = chroma_mean / np.linalg.norm(chroma_mean)

    best_key = None
    best_corr = -999

    # Sprawdzamy wszystkie 12 tonów i 2 tryby (major/minor)
    for shift in range(12):
        # Rotacja wzorca (major)
        major_profile_rot = np.roll(MAJOR_PROFILE, shift)
        minor_profile_rot = np.roll(MINOR_PROFILE, shift)

        major_profile_norm = major_profile_rot / np.linalg.norm(major_profile_rot)
        minor_profile_norm = minor_profile_rot / np.linalg.norm(minor_profile_rot)

        # Korelacja z wektorem chroma_norm
        corr_major = np.dot(chroma_norm, major_profile_norm)
        corr_minor = np.dot(chroma_norm, minor_profile_norm)

        # Czy trafiliśmy lepszą korelację?
        if corr_major > best_corr:
            best_corr = corr_major
            best_key = f"{NOTES[shift]} major"

        if corr_minor > best_corr:
            best_corr = corr_minor
            best_key = f"{NOTES[shift]} minor"

    return best_key
