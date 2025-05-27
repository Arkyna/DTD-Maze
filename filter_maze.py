import cv2
import mediapipe as mp
import speech_recognition as sr
import numpy as np
from PIL import Image, ImageSequence # Pastikan ImageSequence diimpor
import threading
import time
import os

# --------------- KONFIGURASI & SETUP AWAL ---------------
# Path ke aset
ASSETS_DIR = "assets"
DENIS_DIAM_PATH = os.path.join(ASSETS_DIR, "denis_diam.png") # Sebagai fallback jika GIF gagal
LABIRIN_PATH = os.path.join(ASSETS_DIR, "labirin.png")
HINT_LABIRIN_PATH = os.path.join(ASSETS_DIR, "hint_labirin.png")
DENIS_GIF_PATH = os.path.join(ASSETS_DIR, "denis_bergerak.gif") # Path ke file GIF kamu

# Global variables
show_hint = False
hint_display_start_time = 0
HINT_DURATION = 5  # Detik
denis_position = (0, 0)
USE_GIF_DENIS = True # INI PENTING, pastikan True untuk menggunakan GIF

# --------------- MEMUAT ASET GAMBAR ---------------
def load_image_with_alpha(path):
    try:
        img = Image.open(path).convert("RGBA")
        return img
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {path}")
        return None
    except Exception as e:
        print(f"Error memuat gambar {path}: {e}")
        return None

# Memuat aset utama
denis_diam_pil = load_image_with_alpha(DENIS_DIAM_PATH) # Tetap muat sebagai fallback
labirin_pil = load_image_with_alpha(LABIRIN_PATH)
hint_labirin_pil = load_image_with_alpha(HINT_LABIRIN_PATH)

# Memuat Denis bergerak (FOKUS PADA GIF)
denis_frames_pil = []
current_denis_frame_idx = 0
last_denis_frame_time = time.time()
# DENIS_FRAME_DELAY akan diambil dari properti GIF jika ada, atau default
DENIS_DEFAULT_FRAME_DELAY = 0.1 # Detik antar frame Denis jika GIF tidak punya info durasi

gif_frame_durations = [] # Untuk menyimpan durasi setiap frame GIF

if USE_GIF_DENIS:
    try:
        denis_gif = Image.open(DENIS_GIF_PATH)
        # Iterasi melalui setiap frame dalam GIF
        for i, frame_pil_raw in enumerate(ImageSequence.Iterator(denis_gif)):
            # Konversi setiap frame ke RGBA untuk konsistensi alpha channel
            denis_frames_pil.append(frame_pil_raw.convert("RGBA"))
            # Coba dapatkan durasi frame dari info GIF (dalam milidetik)
            try:
                duration_ms = frame_pil_raw.info.get('duration', DENIS_DEFAULT_FRAME_DELAY * 1000)
                gif_frame_durations.append(duration_ms / 1000.0) # Konversi ke detik
            except:
                gif_frame_durations.append(DENIS_DEFAULT_FRAME_DELAY)

        if not denis_frames_pil:
            print("Warning: Gagal memuat frame dari GIF. Cek file GIF.")
            USE_GIF_DENIS = False # Fallback ke Denis diam jika GIF kosong
        else:
            print(f"Berhasil memuat {len(denis_frames_pil)} frame dari GIF.")
    except FileNotFoundError:
        print(f"Error: File GIF Denis tidak ditemukan di {DENIS_GIF_PATH}.")
        USE_GIF_DENIS = False # Fallback
    except Exception as e:
        print(f"Error memuat GIF Denis: {e}.")
        USE_GIF_DENIS = False # Fallback
else:
    # Jika USE_GIF_DENIS False, kita tidak melakukan apa-apa di sini
    # karena tidak ada sequence PNG yang ingin dimuat lagi.
    print("Mode GIF dinonaktifkan, akan mencoba menggunakan Denis diam jika ada.")

# Fallback jika GIF gagal dan Denis diam ada
if not denis_frames_pil and denis_diam_pil:
    denis_frames_pil = [denis_diam_pil] # Anggap sebagai animasi 1 frame
    gif_frame_durations = [DENIS_DEFAULT_FRAME_DELAY] # Beri durasi default
    print("Menggunakan Denis diam sebagai fallback animasi.")
elif not denis_frames_pil and not denis_diam_pil:
     print("CRITICAL: Tidak ada aset Denis (GIF atau diam) yang bisa dimuat.")


# Fungsi untuk overlay gambar PIL ke frame OpenCV (menangani alpha channel)
# (Sama seperti sebelumnya, tidak perlu diubah)
def overlay_image_alpha(background_cv, overlay_pil, x, y):
    if overlay_pil is None:
        return background_cv
    overlay_cv = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGBA2BGRA)
    h, w = overlay_cv.shape[:2]
    bh, bw = background_cv.shape[:2]
    x_start, y_start = x, y
    x_end, y_end = x + w, y + h
    overlay_x_start, overlay_y_start = 0, 0
    if x_start < 0:
        overlay_x_start = -x_start
        x_start = 0
    if y_start < 0:
        overlay_y_start = -y_start
        y_start = 0
    if x_end > bw:
        x_end = bw
    if y_end > bh:
        y_end = bh
    overlay_w = x_end - x_start
    overlay_h = y_end - y_start
    if overlay_w <= 0 or overlay_h <= 0:
        return background_cv
    background_roi = background_cv[y_start:y_end, x_start:x_end]
    overlay_roi = overlay_cv[overlay_y_start:overlay_y_start + overlay_h,
                             overlay_x_start:overlay_x_start + overlay_w]
    alpha = overlay_roi[:, :, 3] / 255.0
    alpha_inv = 1.0 - alpha
    for c in range(0, 3):
        background_roi[:, :, c] = (alpha * overlay_roi[:, :, c] +
                                   alpha_inv * background_roi[:, :, c])
    return background_cv

# --------------- MEDIAPIPE FACE MESH ---------------
# (Sama seperti sebelumnya)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --------------- SPEECH RECOGNITION ---------------
# (Sama seperti sebelumnya)
recognizer = sr.Recognizer()
microphone = sr.Microphone()
def listen_for_command():
    global show_hint, hint_display_start_time
    with microphone as source:
        print("Mendengarkan perintah 'dit tolongin dit'...")
        while True:
            try:
                # recognizer.adjust_for_ambient_noise(source, duration=0.5) # Coba aktifkan jika perlu
                audio = recognizer.listen(source, phrase_time_limit=5)
                command = recognizer.recognize_google(audio, language="id-ID").lower()
                print(f"Diterima: {command}")
                if "dit tolongin dit" in command:
                    print("Hint diminta!")
                    show_hint = True
                    hint_display_start_time = time.time()
            except sr.WaitTimeoutError: pass
            except sr.UnknownValueError: pass
            except sr.RequestError as e:
                print(f"Error dari layanan Google Speech Recognition; {e}")
                time.sleep(2)
            except Exception as e:
                print(f"Error pada speech recognition: {e}")
                break
speech_thread = threading.Thread(target=listen_for_command, daemon=True)
speech_thread.start()

# --------------- LOOP UTAMA OPENCV ---------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
    frame_height, frame_width = frame.shape[:2]
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Gambar labirin (atau background lain)
    if labirin_pil:
        labirin_resized_pil = labirin_pil.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        current_display_frame = overlay_image_alpha(frame.copy(), labirin_resized_pil, 0, 0)
    else:
        current_display_frame = frame.copy()

    # Gambar hint jika aktif
    if show_hint:
        if time.time() - hint_display_start_time < HINT_DURATION:
            if hint_labirin_pil:
                hint_resized_pil = hint_labirin_pil.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
                current_display_frame = overlay_image_alpha(current_display_frame, hint_resized_pil, 0, 0)
        else:
            show_hint = False

    # Deteksi wajah dan update posisi Denis
    denis_x_center, denis_y_center = frame_width // 2, frame_height // 2 # Default jika wajah tidak terdeteksi
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1] # Landmark hidung
            denis_x_center = int(nose_tip.x * frame_width)
            denis_y_center = int(nose_tip.y * frame_height)
            denis_position = (denis_x_center, denis_y_center) # Simpan untuk referensi jika perlu

    # Pilih frame Denis yang akan ditampilkan (animasi GIF)
    denis_to_draw_pil = None
    if denis_frames_pil: # Pastikan ada frame yang dimuat
        # Tentukan delay untuk frame saat ini
        current_frame_delay = gif_frame_durations[current_denis_frame_idx] if gif_frame_durations else DENIS_DEFAULT_FRAME_DELAY

        if time.time() - last_denis_frame_time > current_frame_delay:
            current_denis_frame_idx = (current_denis_frame_idx + 1) % len(denis_frames_pil)
            last_denis_frame_time = time.time()
        denis_to_draw_pil = denis_frames_pil[current_denis_frame_idx]

    # Gambar Denis
    if denis_to_draw_pil:
        denis_w_target = 80
        original_w, original_h = denis_to_draw_pil.size
        if original_w == 0 : original_w = 1 # hindari division by zero
        denis_h_target = int((original_h / original_w) * denis_w_target)
        if denis_h_target == 0: denis_h_target = denis_w_target # Jaga-jaga jika original_h 0
        
        denis_resized_pil = denis_to_draw_pil.resize((denis_w_target, denis_h_target), Image.Resampling.LANCZOS)

        # Gunakan denis_x_center, denis_y_center yang sudah diupdate
        draw_x = denis_x_center - denis_w_target // 2
        draw_y = denis_y_center - denis_h_target - 10 # Sedikit di atas titik acuan (hidung)
        current_display_frame = overlay_image_alpha(current_display_frame, denis_resized_pil, draw_x, draw_y)

    cv2.imshow('Dit Tolongin Dit Maze Filter', current_display_frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --------------- CLEANUP ---------------
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
print("Filter ditutup.")