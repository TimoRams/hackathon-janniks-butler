import os
import pyaudio
import wave
import whisper
import pathlib
from pathlib import Path
import tempfile
import time
from gpiozero import Button, GPIOZeroError
import threading
from paho.mqtt import client as mqtt_client
# --- Konfiguration ---
MODEL_SIZE = "turbo"  # "tiny", "base", "small", "medium", "large". "base" ist ein guter Kompromiss für den Pi.
LANGUAGE = "de"
SAMPLE_RATE = 16000  # Whisper erwartet 16kHz
NOTES_FILENAME = "notizen.txt"
BUTTON_PIN = 14  # GPIO-Pin für den Button

#MQTT-Konfiguration
MQTT_BROKER = "172.16.1.186"
MQTT_PORT = 1883
MQTT_TOPIC = "notes/textnachrichten/text"
MQTT_client_id = f'python-mqtt-Speech2Text'

# PyAudio-spezifische Konfiguration
FORMAT = pyaudio.paInt16
CHANNELS = 1
CHUNK = 1024

TEMP_DIR = Path(tempfile.gettempdir())
RECORDING_FILENAME = str(TEMP_DIR / "aufnahme.wav")
print(f"Temporäre Audiodatei wird gespeichert unter: {RECORDING_FILENAME}")

# Globale Variable für den Button und den Fallback-Modus
button = None
is_gpio_available = False

# --- Initialisierung ---
print(f"Lade das Whisper-Modell '{MODEL_SIZE}'...")
try:
    # Überprüfen, ob das Modell bereits heruntergeladen wurde, um Zeit zu sparen
    model_path = os.path.expanduser(f"~/.cache/whisper/{MODEL_SIZE}.pt")
    if not os.path.exists(model_path):
        print("Modell wird heruntergeladen. Dies kann einige Minuten dauern...")
    model = whisper.load_model(MODEL_SIZE)
    print("Modell erfolgreich geladen.")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    print("Stellen Sie sicher, dass 'ffmpeg' auf Ihrem Raspberry Pi installiert ist.")
    print("Installationsbefehl: sudo apt-get update && sudo apt-get install ffmpeg")
    exit()

def setup_button():
    """Konfiguriert den GPIO-Button mit gpiozero."""
    global button, is_gpio_available
    try:
        # pull_up=True, da dies die häufigste und robusteste Konfiguration ist.
        # Der Pin ist HIGH, wenn der Button nicht gedrückt ist, und LOW, wenn er gedrückt wird.
        button = Button(BUTTON_PIN, pull_up=True)
        print(f"Button an GPIO-Pin {BUTTON_PIN} erfolgreich initialisiert (pull_up=True).")
        is_gpio_available = True
    except GPIOZeroError as e:
        print(f"GPIO-Fehler (gpiozero): {e}")
        print("Stellen Sie sicher, dass das Skript auf einem Raspberry Pi läuft und die Berechtigungen korrekt sind.")
        print("FALLBACK: Verwende Terminal (Enter) zum Starten/Stoppen der Aufnahme.")
        is_gpio_available = False

def record_audio():
    """Nimmt Audio auf, entweder per GPIO-Button oder per Terminal-Eingabe."""
    print("-" * 50)

    if is_gpio_available:
        print("Drücke und halte den Button, um die Aufnahme zu starten. Lasse los, um zu beenden.")
        # Warten, bis der Button gedrückt wird (Pin geht auf LOW)
        button.wait_for_press()
        print("Aufnahme startet...")
    else:
        input("Drücke ENTER, um die Aufnahme zu starten...")
        print("Aufnahme startet... Drücke ENTER erneut, um zu stoppen.")

    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=SAMPLE_RATE, input=True,
                        frames_per_buffer=CHUNK)
    
    frames = []
    
    try:
        if is_gpio_available:
            # Lese Daten, solange der Button gedrückt ist (Pin ist LOW)
            while button.is_pressed:
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)
        else:
            # Terminal-Fallback: Aufnahme in einem separaten Thread
            stop_event = threading.Event()
            def _record_loop():
                while not stop_event.is_set():
                    data = stream.read(CHUNK, exception_on_overflow=False)
                    frames.append(data)
            
            rec_thread = threading.Thread(target=_record_loop)
            rec_thread.start()
            input() # Wartet auf das zweite Enter
            stop_event.set()
            rec_thread.join()

    except Exception as e:
        print(f"Fehler während der Aufnahme: {e}")

    print("Aufnahme beendet.")

    # Stream und PyAudio schließen
    stream.stop_stream()
    stream.close()
    audio.terminate()

    if not frames:
        return None, None
    
    p = pyaudio.PyAudio()
    sample_width = p.get_sample_size(FORMAT)
    p.terminate()
    
    return frames, sample_width

def save_audio(filename, frames, sample_width):
    """Speichert die Aufnahme als WAV-Datei."""
    if not frames:
        print("Keine Frames zum Speichern.")
        return

    try:
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(sample_width)
            wf.setframerate(SAMPLE_RATE)
            wf.writeframes(b''.join(frames))
        print(f"Aufnahme temporär gespeichert als '{filename}'.")
    except Exception as e:
        print(f"Fehler beim Speichern der Audiodatei: {e}")


def transcribe_audio(filename):
    """Transkribiert die Audio-Datei mit Whisper."""
    if not os.path.exists(filename):
        return "Fehler: Audiodatei nicht gefunden."

    print("Transkription wird gestartet (dies kann einen Moment dauern)...")
    try:
        result = model.transcribe(filename, language=LANGUAGE, fp16=False)
        text = result["text"]
        print("Transkription erfolgreich.")
        return text
    except Exception as e:
        print(f"Fehler bei der Transkription: {e}")
        return ""

def save_note(text):
    """Speichert den transkribierten Text in der Notizdatei."""
    if not text:
        print("Kein Text zum Speichern.")
        return

    try:
        with open(NOTES_FILENAME, "a", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Notiz erfolgreich in '{NOTES_FILENAME}' gespeichert.")
    except Exception as e:
        print(f"Fehler beim Speichern der Notiz: {e}")

def connect_mqtt():
    def on_connect(client, userdata, flags, rc, properties):
        if rc == 0:
            print("Mit MQTT-Broker verbunden.")
        else:
            print(f"Verbindungsfehler mit MQTT-Broker, Rückgabecode {rc}")

    client = mqtt_client.Client(mqtt_client.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    try:
        client.connect(MQTT_BROKER, MQTT_PORT)
    except Exception as e:
        print(f"Fehler bei der Verbindung zum MQTT-Broker: {e}")
        return None
    return client
        
def publish_mqtt(client, topic,msg):
    result = client.publish(topic, msg)
    if result[0] == 0:
        print(f"Nachricht erfolgreich an Topic '{topic}' gesendet.")
    else:
        print(f"Fehler beim Senden der Nachricht an Topic '{topic}'.")
    

def main():
    """Hauptfunktion des Programms."""
    mqtt_client_instance = connect_mqtt()
    mqtt_client_instance.loop_start() if mqtt_client_instance else None
    try:
        setup_button()
        print("\nWillkommen beim Sprachnotiz-Assistenten!")
        print("Drücken Sie STRG+C, um das Programm jederzeit zu beenden.")

        while True:
            # 1. Audio aufnehmen
            audio_frames, sample_width = record_audio()
            if audio_frames is None:
                print("Keine Audiodaten aufgenommen.")
                continue

            # 2. Audio speichern
            try:
                print("Speichere die Aufnahme...")
                save_audio(RECORDING_FILENAME, audio_frames, sample_width)
            except Exception as e:
                print(f"Fehler beim Speichern der Audiodatei: {e}")
                continue
                
            # 3. Audio transkribieren
            transcribed_text = transcribe_audio(RECORDING_FILENAME)
            print("-" * 50)
            print(f"Erkannter Text: {transcribed_text}")
            print("-" * 50)
            
            # 4. Notiz speichern
            save_note(transcribed_text)
            
            # 5. Text per MQTT senden
            publish_mqtt(mqtt_client_instance, MQTT_TOPIC, transcribed_text) if mqtt_client_instance else None

            # Temporäre Audiodatei löschen
            if os.path.exists(RECORDING_FILENAME):
                os.remove(RECORDING_FILENAME)

    except KeyboardInterrupt:
        print("\nProgramm wird beendet. Auf Wiedersehen!")
    except Exception as e:
        print(f"Ein unerwarteter Fehler ist aufgetreten: {e}")
    finally:
        # gpiozero räumt automatisch auf, kein GPIO.cleanup() nötig.
        print("Programm beendet.")
        mqtt_client_instance.loop_stop() if mqtt_client_instance else None

if __name__ == "__main__":
    main()