import os
import time
import json
import cv2
import numpy as np
from datetime import datetime
import paho.mqtt.client as mqtt
import face_recognition
import signal

# ================= KONFIGURATION =================

VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 10

print(f"📐 Video-Auflösung: {VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS}fps")

def open_camera():
    for idx in [0, 1, 2]:
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, VIDEO_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, VIDEO_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, VIDEO_FPS)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ Kamera gefunden: /dev/video{idx} ({VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS}fps)")
                return cap
            cap.release()
    
    raise RuntimeError("❌ Keine Kamera gefunden")

def ensure_bgr(frame):
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.ndim == 3 and frame.shape[2] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame

def create_tracker():
    chain = []
    if hasattr(cv2, "legacy"):
        chain += [
            lambda: cv2.legacy.TrackerCSRT_create(),
            lambda: cv2.legacy.TrackerKCF_create(),
            lambda: cv2.legacy.TrackerMOSSE_create(),
        ]
    chain += [
        lambda: cv2.TrackerCSRT_create() if hasattr(cv2, "TrackerCSRT_create") else (_ for _ in ()).throw(RuntimeError()),
        lambda: cv2.TrackerKCF_create() if hasattr(cv2, "TrackerKCF_create") else (_ for _ in ()).throw(RuntimeError()),
    ]
    for make in chain:
        try:
            return make()
        except Exception:
            continue
    raise RuntimeError("Kein geeigneter OpenCV-Tracker verfügbar.")

os.environ['QT_QPA_PLATFORM'] = 'xcb'
os.environ['DISPLAY'] = ':0'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

LEARN_NEW_FACES = False
SHOW_VISUAL_UI = True
MAX_MISSES = 5
RECOGNITION_INTERVAL = 15
RESIZE_THRESHOLD = 0.3
MIN_ENCODING_QUALITY = 0.08
MAX_ENCODINGS_PER_PERSON = 2
ENCODING_UPDATE_THRESHOLD = 0.3
FRAME_SKIP = 1

DATABASE_FILE = "face_database.json"
MQTT_BROKER = "172.16.1.186"
MQTT_PORT = 1883
MQTT_TOPIC = "notes/textnachrichten/gesicht"
MQTT_CLIENT_ID = "python-mqtt-timo"

known_encodings = []
known_ids = []
person_names = {}
next_id = 1
tracked_objects = []
frame_count = 0
encoding_qualities = []
last_main_subject_print = 0
MAIN_SUBJECT_INTERVAL = 3.0
mqtt_client = None

def calculate_overlap(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    x_left = max(x1, x2); y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2); y_bottom = min(y1 + h1, y2 + h2)
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1; area2 = w2 * h2
    return intersection / (area1 + area2 - intersection)

def get_face_quality(face_rgb):
    if face_rgb.size == 0:
        return 0.0
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(sharpness / 500.0, 1.0)
    brightness_std = np.std(gray)
    brightness_score = min(brightness_std / 64.0, 1.0)
    size = face_rgb.shape[0] * face_rgb.shape[1]
    size_score = min(size / 10000.0, 1.0)
    quality = (sharpness_score * 0.5) + (brightness_score * 0.3) + (size_score * 0.2)
    return min(quality, 1.0)

def save_database():
    try:
        with open(DATABASE_FILE, 'w', encoding='utf-8') as f:
            header = {"next_id": next_id, "last_saved": datetime.now().isoformat()}
            f.write(json.dumps(header, separators=(',', ':'), ensure_ascii=False) + '\n')
            for i, person_id in enumerate(known_ids):
                person_data = {
                    'id': person_id,
                    'name': person_names.get(person_id, ""),
                    'encodings': [enc.tolist() for enc in known_encodings[i]],
                    'qualities': encoding_qualities[i]
                }
                f.write(json.dumps(person_data, separators=(',', ':'), ensure_ascii=False) + '\n')
        print(f"Datenbank gespeichert: {len(known_ids)} Personen")
    except Exception as e:
        print(f"Fehler beim Speichern: {e}")

def load_database():
    global known_encodings, known_ids, encoding_qualities, next_id, person_names
    if not os.path.exists(DATABASE_FILE):
        print(f"{DATABASE_FILE} nicht gefunden, starte mit leerer Datenbank")
        return
    try:
        known_encodings, known_ids = [], []
        encoding_qualities = []
        person_names = {}
        with open(DATABASE_FILE, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        if not lines:
            return
        header = json.loads(lines[0].strip())
        next_id = header.get('next_id', 1)
        for line in lines[1:]:
            if line.strip():
                person = json.loads(line.strip())
                known_ids.append(person['id'])
                known_encodings.append([np.array(enc) for enc in person['encodings']])
                encoding_qualities.append(person['qualities'])
                if person['name'].strip():
                    person_names[person['id']] = person['name']
        print(f"Datenbank geladen: {len(known_ids)} Personen")
        if person_names:
            print("Namen gefunden:")
            for pid, name in person_names.items():
                print(f"  ID {pid}: {name}")
    except Exception as e:
        print(f"Fehler beim Laden: {e}")

def get_person_display_name(pid):
    if pid == -1:
        return "Unbekannt"
    if pid in person_names:
        return f"{person_names[pid]} (ID {pid})"
    return f"ID {pid}"

def find_best_encoding_match(target_encoding, person_encodings, person_ids):
    best_person_id = None
    best_distance = float('inf')
    for pid, enc_list in zip(person_ids, person_encodings):
        if not enc_list:
            continue
        distances = face_recognition.face_distance(enc_list, target_encoding)
        min_distance = np.min(distances) if len(distances) > 0 else float('inf')
        if min_distance < best_distance:
            best_distance = min_distance
            best_person_id = pid
    return best_person_id if best_distance < 0.5 else None

def setup_mqtt():
    global mqtt_client
    try:
        mqtt_client = mqtt.Client(client_id=MQTT_CLIENT_ID, protocol=mqtt.MQTTv311)
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
        print(f"✅ MQTT Client verbunden mit {MQTT_BROKER}")
        return True
    except Exception as e:
        print(f"❌ MQTT Verbindung fehlgeschlagen: {e}")
        mqtt_client = None
        return False

def send_main_subject_mqtt(main_subject):
    if mqtt_client is None:
        return
    try:
        message = "none"
        if main_subject:
            display_name = get_person_display_name(main_subject['id'])
            message = display_name
        mqtt_client.publish(MQTT_TOPIC, message)
        print(f"📤 MQTT: '{message}' gesendet")
    except Exception as e:
        print(f"❌ MQTT Send Fehler: {e}")

def get_main_subject(frame, tracked_objects):
    if not tracked_objects:
        return None
    frame_center_x = frame.shape[1] // 2
    frame_center_y = frame.shape[0] // 2
    best_subject, best_score = None, 0
    for obj in tracked_objects:
        x, y, w, h = obj['last_box']
        cx = x + w // 2; cy = y + h // 2
        dist_center = np.sqrt(((cx - frame_center_x) / frame.shape[1]) ** 2 +
                              ((cy - frame_center_y) / frame.shape[0]) ** 2)
        area = w * h; max_area = frame.shape[0] * frame.shape[1]
        size_score = area / max_area
        stability = min(obj.get('frames_tracked', 0) / 100.0, 1.0)
        score = (1.0 - dist_center) * 0.4 + size_score * 0.4 + stability * 0.2
        if score > best_score:
            best_score = score
            best_subject = obj
    return best_subject

load_database()
setup_mqtt()

try:
    cap = open_camera()
except Exception as e:
    print(f"❌ Konnte keine Kamera öffnen: {e}")
    save_database()
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()
        print("🔌 MQTT Client getrennt")
    raise SystemExit(1)

tracked_objects = []
frame_count = 0
last_main_subject_print = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Kein Frame empfangen—beende.")
        break
    frame = ensure_bgr(frame)
    frame_count += 1

    skip_heavy_processing = frame_count % FRAME_SKIP != 0
    current_boxes = []
    survived_trackers = []

    # Tracker updaten
    for obj in tracked_objects:
        tracker = obj['tracker']
        try:
            success, box = tracker.update(frame)
            if success:
                x, y, w, h = [int(v) for v in box]
                if x >= 0 and y >= 0 and x + w <= frame.shape[1] and y + h <= frame.shape[0] and w > 15 and h > 15:
                    new_box = (x, y, w, h)
                    prev_box = obj['last_box']
                    area_prev = max(1, prev_box[2] * prev_box[3])
                    area_new  = max(1, w * h)
                    ratio = area_new / area_prev
                    if ratio < (1 - RESIZE_THRESHOLD) or ratio > (1 + RESIZE_THRESHOLD):
                        try:
                            new_tracker = create_tracker()
                            new_tracker.init(frame, new_box)
                            obj['tracker'] = new_tracker
                        except Exception:
                            pass
                    obj['misses'] = 0
                    obj['last_box'] = new_box
                    obj['frames_tracked'] = obj.get('frames_tracked', 0) + 1
                    survived_trackers.append(obj)
                    current_boxes.append(new_box)
                    confidence = obj.get('confidence', 0.5)
                    stability = min(obj['frames_tracked'] / 30.0, 1.0)
                    color_intensity = int(255 * (confidence + stability) / 2)
                    color = (0, color_intensity, 255 - color_intensity)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    display_name = get_person_display_name(obj['id'])
                    cv2.putText(frame, f"{display_name} ({confidence:.2f})", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                else:
                    obj['misses'] += 1
            else:
                obj['misses'] += 1
        except Exception as e:
            print(f"❌ Tracker-Update Fehler: {e}")
            obj['misses'] += 1

    tracked_objects = [obj for obj in survived_trackers if obj['misses'] < MAX_MISSES]

    if frame_count % RECOGNITION_INTERVAL == 0 and not skip_heavy_processing:
        small_frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(20, 20),
            maxSize=(120, 120)
        )
        
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240
        faces_scaled = [(int(x*scale_x), int(y*scale_y), int(w*scale_x), int(h*scale_y)) for x, y, w, h in faces]
        
        detected_face_boxes = [tuple(f) for f in faces_scaled]
        used_faces = [False] * len(detected_face_boxes)
        
        print(f"🔍 Gesichter erkannt: {len(detected_face_boxes)}")

        for obj in tracked_objects:
            best_idx = -1
            best_overlap = 0.3
            for i, fb in enumerate(detected_face_boxes):
                if not used_faces[i]:
                    ov = calculate_overlap(obj['last_box'], fb)
                    if ov > best_overlap:
                        best_overlap = ov
                        best_idx = i
            if best_idx != -1:
                new_box = detected_face_boxes[best_idx]
                try:
                    new_tracker = create_tracker()
                    new_tracker.init(frame, new_box)
                    obj['tracker'] = new_tracker
                    obj['last_box'] = new_box
                    obj['misses'] = 0
                    used_faces[best_idx] = True
                except Exception:
                    pass

        # Neue Gesichter
        for i, face_box in enumerate(detected_face_boxes):
            if not used_faces[i]:
                x, y, w, h = face_box
                # Validiere Bounding Box
                if w < 30 or h < 30 or x < 0 or y < 0 or x+w > frame.shape[1] or y+h > frame.shape[0]:
                    print(f"❌ Ungültige Bounding Box: ({x}, {y}, {w}, {h})")
                    continue
                    
                try:
                    face_rgb = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2RGB)
                    encodings = face_recognition.face_encodings(face_rgb)
                    if encodings:
                        enc = encodings[0]
                        quality = get_face_quality(face_rgb)
                        print(f"🎯 Gesicht gefunden: Quality={quality:.3f}, Min={MIN_ENCODING_QUALITY}")
                        
                        if quality >= MIN_ENCODING_QUALITY:
                            matched_person_id = find_best_encoding_match(enc, known_encodings, known_ids)
                            if matched_person_id is not None:
                                face_id = matched_person_id
                                print(f"✅ Person erkannt: ID {face_id} ({get_person_display_name(face_id)})")
                                if quality > ENCODING_UPDATE_THRESHOLD:
                                    person_idx = known_ids.index(face_id)
                                    if len(known_encodings[person_idx]) < MAX_ENCODINGS_PER_PERSON:
                                        known_encodings[person_idx].append(enc)
                                        encoding_qualities[person_idx].append(quality)
                                    else:
                                        worst_idx = int(np.argmin(encoding_qualities[person_idx]))
                                        if quality > encoding_qualities[person_idx][worst_idx]:
                                            known_encodings[person_idx][worst_idx] = enc
                                            encoding_qualities[person_idx][worst_idx] = quality
                            else:
                                if LEARN_NEW_FACES:
                                    face_id = next_id
                                    known_encodings.append([enc])
                                    known_ids.append(face_id)
                                    encoding_qualities.append([quality])
                                    next_id += 1
                                    save_database()
                                    print(f"📝 Neue Person gelernt: ID {face_id}")
                                else:
                                    face_id = -1
                                    print(f"❓ Unbekannte Person (Lernen deaktiviert)")
                            
                            # Tracker mit verbesserter Initialisierung
                            try:
                                tracker = create_tracker()
                                success = tracker.init(frame, (x, y, w, h))
                                if success:
                                    tracked_objects.append({
                                        'tracker': tracker,
                                        'id': face_id,
                                        'misses': 0,
                                        'last_box': (x, y, w, h),
                                        'frames_tracked': 0,
                                        'confidence': quality
                                    })
                                    print(f"🎯 Tracker initialisiert für ID {face_id}")
                                else:
                                    print(f"❌ Tracker-Initialisierung fehlgeschlagen für ID {face_id}")
                            except Exception as te:
                                print(f"❌ Tracker-Fehler: {te}")
                        else:
                            print(f"⚠️ Gesicht zu schlecht: Quality {quality:.3f} < {MIN_ENCODING_QUALITY}")
                    else:
                        print(f"⚠️ Keine Face-Encodings gefunden")
                except Exception as e:
                    print(f"❌ Face recognition error: {e}")

    now = time.time()
    if now - last_main_subject_print >= MAIN_SUBJECT_INTERVAL:
        print(f"🔄 Tracked Objects: {len(tracked_objects)}")
        for i, obj in enumerate(tracked_objects):
            print(f"   [{i}] ID: {obj['id']}, Frames: {obj.get('frames_tracked', 0)}, Misses: {obj['misses']}")
        
        main_subject = get_main_subject(frame, tracked_objects)
        if main_subject:
            display_name = get_person_display_name(main_subject['id'])
            confidence = main_subject.get('confidence', 0.0)
            print(f">>> MAIN SUBJECT: {display_name} (Confidence: {confidence:.2f})")
        else:
            print(">>> MAIN SUBJECT: Niemand erkannt")
        send_main_subject_mqtt(main_subject)
        last_main_subject_print = now

    if SHOW_VISUAL_UI:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key >= ord("1") and key <= ord("9"):
            selected_id = key - ord("0")
            if selected_id in known_ids:
                print(f"\nName für Person ID {selected_id} eingeben:")
                print("Aktueller Name:", person_names.get(selected_id, "Unbenannt"))
                print("Drücke 'n' um Namen zu ändern")
        elif key == ord("n"):
            print("\nVerfügbare Personen:")
            for pid in known_ids:
                print(f"  ID {pid}: {person_names.get(pid, 'Unbenannt')}")
            print("Um Namen zu setzen: python assign_name.py <ID> <Name>")
        elif key == ord("s"):
            save_database()
            print("Daten gespeichert!")
        
        try:
            cv2.imshow("Face Re-Identification", frame)
        except Exception as e:
            print(f"⚠️ Display-Fehler (läuft headless): {e}")
            SHOW_VISUAL_UI = False
    else:
        def signal_handler(sig, frame):
            print("\n🛑 Beende Programm...")
            raise KeyboardInterrupt
        signal.signal(signal.SIGINT, signal_handler)
        time.sleep(0.01)

save_database()
if mqtt_client:
    mqtt_client.loop_stop()
    mqtt_client.disconnect()
    print("🔌 MQTT Client getrennt")
cap.release()
cv2.destroyAllWindows()
