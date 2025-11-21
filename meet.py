import speech_recognition as sr
from deep_translator import GoogleTranslator
from langdetect import detect
import pyaudio
import wave
import threading
import queue
import time
import os
import numpy as np
import subprocess


# Configuration
CHUNK = 1024  # Audio chunk size
FORMAT = pyaudio.paInt16  # Audio format
CHANNELS = 1  # Mono audio
RATE = 44100  # Sample rate
MAX_CHUNK_DURATION = 1.5  # Max audio chunk duration
OVERLAP_DURATION = 0.5  # Keep 0.5 seconds of previous audio
SILENCE_THRESHOLD = 1.0  # Silence duration to trigger early processing
MAX_QUEUE_SIZE = 10  # Larger queue to prevent dropping audio

# Generate unique filenames with timestamp
TIMESTAMP = time.strftime("%Y-%m-%d_%H-%M-%S")
TRANSCRIPT_FILE = f"transcript_{TIMESTAMP}.txt"
CONVERSATION_FILE = f"conversation_{TIMESTAMP}.txt"

# Initialize global components
recognizer = sr.Recognizer()
recognizer.energy_threshold = 200  # Lower threshold for better sensitivity
recognizer.dynamic_energy_threshold = True  # Adapt to background noise
audio_queue = queue.Queue(maxsize=MAX_QUEUE_SIZE)
stop_event = threading.Event()
last_transcription = ""  # Track last transcription to avoid duplicates
recent_transcriptions = []  # Buffer recent transcriptions for context

def save_to_files(original, language, translation):
    """Save transcription and translation to both files."""
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Save to transcript file
    with open(TRANSCRIPT_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] Original ({language}): {original}\n")
        f.write(f"[{timestamp}] Translation: {translation}\n\n")
    
    # Save to conversation file
    with open(CONVERSATION_FILE, 'a', encoding='utf-8') as f:
        f.write(f"[{timestamp}] Caption: {original}\n")
        f.write(f"[{timestamp}] Detected Language: {language}\n")
        f.write(f"[{timestamp}] Translation: {translation}\n\n")
    
    print(f"Saved to {TRANSCRIPT_FILE} and {CONVERSATION_FILE}")

def detect_language(text):
    """Detect language of the input text with fallback."""
    try:
        lang = detect(text)
        return lang if lang in ['ar', 'en'] else 'en'  # Restrict to Arabic or English
    except:
        return 'en'  # Default to English if detection fails

def normalize_text(text):
    """Normalize text for duplicate checking."""
    return text.lower().strip()

def process_audio():
    """Process audio from the queue for transcription and translation."""
    global last_transcription, recent_transcriptions
    while not stop_event.is_set():
        try:
            audio_data = audio_queue.get(timeout=0.1)
            if not audio_data:
                continue

            with wave.open("temp.wav", 'wb') as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(pyaudio.PyAudio().get_sample_size(FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(audio_data)

            with sr.AudioFile("temp.wav") as source:
                audio = recognizer.record(source)
                try:
                    # Recognize speech with timing
                    start_time = time.time()
                    text = recognizer.recognize_google(audio, language='auto')
                    recognition_time = time.time() - start_time

                    normalized_text = normalize_text(text)
                    if normalized_text == normalize_text(last_transcription):
                        continue  # Skip exact duplicates
                    last_transcription = text

                    # Add to recent transcriptions (keep last 2 for context)
                    recent_transcriptions.append(text)
                    if len(recent_transcriptions) > 2:
                        recent_transcriptions.pop(0)

                    # Combine recent transcriptions for context
                    combined_text = " ".join(recent_transcriptions)

                    # Detect language
                    lang = detect_language(combined_text)
                    lang_name = 'Arabic' if lang == 'ar' else 'English'

                    # Dynamically translate based on detected language
                    start_time = time.time()
                    if lang == 'ar':
                        # Translate Arabic to English
                        translated = GoogleTranslator(source='ar', target='en').translate(combined_text)
                    else:
                        # Translate English to Arabic
                        translated = GoogleTranslator(source='en', target='ar').translate(combined_text)
                    translation_time = time.time() - start_time

                    # Print all results together with timing
                    print(f"\rCaption: {combined_text}")
                    print(f"Detected Language: {lang_name}")
                    print(f"Translation: {translated}")
                    print(f"[Recognition: {recognition_time:.2f}s, Translation: {translation_time:.2f}s]")
                    save_to_files(combined_text, lang_name, translated)

                except sr.UnknownValueError:
                    print("\r[Skipped unclear audio]", end='', flush=True)
                except sr.RequestError as e:
                    print(f"\rRecognition error: {e}", end='', flush=True)
                except Exception as e:
                    print(f"\rError processing audio: {e}", end='', flush=True)

        except queue.Empty:
            continue

def stream_audio():
    """Capture audio stream and put chunks into the queue."""
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Listening for speech... (Speak now)")
    audio_buffer = b''
    overlap_buffer = b''
    chunk_start_time = time.time()
    last_speech_time = time.time()
    speech_detected = False

    while not stop_event.is_set():
        try:
            data = stream.read(CHUNK, exception_on_overflow=False)
            audio_buffer += data

            # Convert audio chunk to numpy array for energy calculation
            audio_array = np.frombuffer(data, dtype=np.int16)
            energy = np.sqrt(np.mean(audio_array**2))

            # Check if energy exceeds threshold (speech detected)
            if energy > recognizer.energy_threshold:
                speech_detected = True
                last_speech_time = time.time()
            else:
                # If silence for 1 second after speech, process chunk
                if speech_detected and (time.time() - last_speech_time >= SILENCE_THRESHOLD):
                    if audio_buffer:
                        chunk_to_process = overlap_buffer + audio_buffer
                        try:
                            audio_queue.put(chunk_to_process, block=False)
                            print(f"\r[Queued {len(chunk_to_process)/RATE:.2f}s audio (silence)]", end='', flush=True)
                        except queue.Full:
                            print("\r[Queue full, audio dropped]", end='', flush=True)

                        # Update overlap and reset
                        overlap_samples = int(OVERLAP_DURATION * RATE * 2)
                        overlap_buffer = audio_buffer[-overlap_samples:] if len(audio_buffer) > overlap_samples else audio_buffer
                        audio_buffer = b''
                        chunk_start_time = time.time()
                        speech_detected = False
                        last_speech_time = time.time()

            # Process audio if max duration reached
            if time.time() - chunk_start_time >= MAX_CHUNK_DURATION:
                if audio_buffer:
                    chunk_to_process = overlap_buffer + audio_buffer
                    try:
                        audio_queue.put(chunk_to_process, block=False)
                        print(f"\r[Queued {len(chunk_to_process)/RATE:.2f}s audio (max)]", end='', flush=True)
                    except queue.Full:
                        print("\r[Queue full, audio dropped]", end='', flush=True)

                    # Update overlap and reset
                    overlap_samples = int(OVERLAP_DURATION * RATE * 2)
                    overlap_buffer = audio_buffer[-overlap_samples:] if len(audio_buffer) > overlap_samples else audio_buffer
                    audio_buffer = b''
                    chunk_start_time = time.time()
                    speech_detected = False
                    last_speech_time = time.time()

        except Exception as e:
            print(f"Error capturing audio: {e}")
            break

    # Process any remaining audio
    if audio_buffer or overlap_buffer:
        try:
            audio_queue.put(overlap_buffer + audio_buffer, block=False)
        except queue.Full:
            pass

    stream.stop_stream()
    stream.close()
    p.terminate()

def main():
    """Main function to start real-time transcription and translation."""
    print("Starting Real-Time Caption System...")
    print(f"Logging to {TRANSCRIPT_FILE} and {CONVERSATION_FILE}")

    # Start audio streaming and processing in separate threads
    audio_thread = threading.Thread(target=stream_audio)
    process_thread = threading.Thread(target=process_audio)

    audio_thread.start()
    process_thread.start()

    try:
        # Run until interrupted (e.g., Ctrl+C)
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping caption system...")
        stop_event.set()

    # Wait for threads to finish
    audio_thread.join()
    process_thread.join()
    print("Caption system stopped. Results saved to", TRANSCRIPT_FILE, "and", CONVERSATION_FILE)

    # After transcription is done, run ai_analysis.py
    print("Triggering AI analysis...")
    subprocess.run(["python", "ai_analysis.py", TRANSCRIPT_FILE], check=True)
    print("AI analysis completed and PDF generated.")

if __name__ == "__main__":
    main()