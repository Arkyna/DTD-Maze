import cv2
import mediapipe as mp
import speech_recognition as sr
import numpy as np
from PIL import Image, ImageSequence
import threading
import time
import os

# --------------- KONFIGURASI & SETUP AWAL ---------------
ASSETS_DIR = "assets"
DENIS_DIAM_PATH = os.path.join(ASSETS_DIR, "denis_diam.png")
LABIRIN_PATH = os.path.join(ASSETS_DIR, "labirin.png")
HINT_LABIRIN_PATH = os.path.join(ASSETS_DIR, "hint_labirin.png")
DENIS_GIF_PATH = os.path.join(ASSETS_DIR, "denis_bergerak.gif")

show_hint = False
hint_display_start_time = 0
HINT_DURATION = 5
USE_GIF_DENIS = True

# BARU: Konfigurasi untuk deteksi tabrakan
WALL_COLOR_THRESHOLD = 60 # Anggap warna di bawah ini (pada skala abu-abu) adalah dinding
DENIS_SPEED = 5  # Piksel per frame, sesuaikan untuk kecepatan yang pas
DENIS_COLLISION_PADDING = 2 # Sedikit padding untuk deteksi agar tidak terlalu mepet

# Global untuk posisi Denis (akan diupdate dengan collision detection)
denis_current_x = 0 # Akan diinisialisasi nanti
denis_current_y = 0 # Akan diinisialisasi nanti
denis_initialized = False # Flag untuk menandai apakah posisi awal Denis sudah diset

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

denis_diam_pil = load_image_with_alpha(DENIS_DIAM_PATH)
labirin_pil_original = load_image_with_alpha(LABIRIN_PATH) # Simpan versi original
hint_labirin_pil_original = load_image_with_alpha(HINT_LABIRIN_PATH) # Simpan versi original

# BARU: Labirin yang akan di-resize sesuai frame, digunakan untuk collision detection
labirin_pil_resized_for_collision = None

denis_frames_pil = []
current_denis_frame_idx = 0
last_denis_frame_time = time.time()
DENIS_DEFAULT_FRAME_DELAY = 0.1
gif_frame_durations = []

if USE_GIF_DENIS:
    try:
        denis_gif = Image.open(DENIS_GIF_PATH)
        for i, frame_pil_raw in enumerate(ImageSequence.Iterator(denis_gif)):
            denis_frames_pil.append(frame_pil_raw.convert("RGBA"))
            try:
                duration_ms = frame_pil_raw.info.get('duration', DENIS_DEFAULT_FRAME_DELAY * 1000)
                gif_frame_durations.append(duration_ms / 1000.0)
            except:
                gif_frame_durations.append(DENIS_DEFAULT_FRAME_DELAY)
        if not denis_frames_pil:
            print("Warning: Gagal memuat frame dari GIF.")
            USE_GIF_DENIS = False
        else:
            print(f"Berhasil memuat {len(denis_frames_pil)} frame dari GIF.")
    except FileNotFoundError:
        print(f"Error: File GIF Denis tidak ditemukan di {DENIS_GIF_PATH}.")
        USE_GIF_DENIS = False
    except Exception as e:
        print(f"Error memuat GIF Denis: {e}.")
        USE_GIF_DENIS = False
else:
    print("Mode GIF dinonaktifkan.")

if not denis_frames_pil and denis_diam_pil:
    denis_frames_pil = [denis_diam_pil]
    gif_frame_durations = [DENIS_DEFAULT_FRAME_DELAY]
    print("Menggunakan Denis diam sebagai fallback animasi.")
elif not denis_frames_pil and not denis_diam_pil:
     print("CRITICAL: Tidak ada aset Denis (GIF atau diam) yang bisa dimuat.")

# --------------- FUNGSI OVERLAY & COLLISION ---------------
def overlay_image_alpha(background_cv, overlay_pil, x, y):
    # (Kode overlay_image_alpha kamu sudah oke, tidak perlu diubah)
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
    alpha_channel = overlay_roi[:, :, 3]
    # Pastikan alpha_channel adalah float antara 0 dan 1
    if alpha_channel.dtype != np.float32 and alpha_channel.dtype != np.float64 :
        alpha = alpha_channel / 255.0
    else:
        alpha = alpha_channel

    # Jika overlay punya bagian transparan, kita perlu blend
    # Jika tidak, kita bisa langsung timpa (tapi blend lebih aman)
    if np.any(alpha < 1.0): # Ada transparansi
        alpha_expanded = np.expand_dims(alpha, axis=2) # Untuk broadcasting
        background_roi[:, :, :3] = (alpha_expanded * overlay_roi[:, :, :3] +
                                    (1 - alpha_expanded) * background_roi[:, :, :3])
    else: # Sepenuhnya opaque, bisa timpa, tapi blend tetap bekerja
        alpha_expanded = np.expand_dims(alpha, axis=2)
        background_roi[:, :, :3] = (alpha_expanded * overlay_roi[:, :, :3] +
                                    (1 - alpha_expanded) * background_roi[:, :, :3])
    # Background_cv sudah dimodifikasi secara in-place
    return background_cv


# BARU: Fungsi untuk mengecek tabrakan
def check_collision(denis_bbox_next, maze_image_pil):
    """
    Mengecek apakah bounding box Denis di posisi berikutnya akan bertabrakan
    dengan dinding pada gambar maze.
    denis_bbox_next: tuple (x_min, y_min, x_max, y_max) untuk posisi Denis berikutnya.
    maze_image_pil: Gambar PIL labirin yang sudah di-resize sesuai frame.
    """
    if maze_image_pil is None:
        return False # Tidak ada labirin, tidak ada tabrakan

    # Kita akan mengecek beberapa titik di dalam bounding box Denis
    # Titik-titik ini harus dalam koordinat maze_image_pil
    x_min, y_min, x_max, y_max = denis_bbox_next

    # Ambil beberapa sampel titik di tepi dan tengah bounding box
    # (bisa disesuaikan untuk akurasi vs performa)
    # Untuk simpelnya, kita cek 4 sudut dan tengah setiap sisi
    check_points_relative = [
        (0, 0), (x_max - x_min, 0),  # Kiri atas, Kanan atas
        (0, y_max - y_min), (x_max - x_min, y_max - y_min), # Kiri bawah, Kanan bawah
        ((x_max - x_min) // 2, 0), # Tengah atas
        ((x_max - x_min) // 2, y_max - y_min), # Tengah bawah
        (0, (y_max - y_min) // 2), # Tengah kiri
        (x_max - x_min, (y_max - y_min) // 2) # Tengah kanan
    ]

    maze_w, maze_h = maze_image_pil.size
    # Konversi labirin ke grayscale untuk deteksi warna dinding yang lebih mudah
    # Lakukan ini sekali saja per frame jika labirin tidak berubah
    maze_gray_pil = maze_image_pil.convert('L')

    for rel_x, rel_y in check_points_relative:
        abs_x = x_min + rel_x
        abs_y = y_min + rel_y

        # Pastikan titik berada dalam batas gambar labirin
        if 0 <= abs_x < maze_w and 0 <= abs_y < maze_h:
            pixel_color = maze_gray_pil.getpixel((abs_x, abs_y))
            # Asumsi dinding berwarna gelap (misal hitam)
            if pixel_color < WALL_COLOR_THRESHOLD:
                # print(f"Collision at ({abs_x}, {abs_y}), color: {pixel_color}")
                return True  # Tabrakan terdeteksi
    return False # Tidak ada tabrakan

# --------------- MEDIAPIPE FACE MESH ---------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --------------- SPEECH RECOGNITION ---------------
recognizer = sr.Recognizer()
microphone = sr.Microphone()
def listen_for_command():
    global show_hint, hint_display_start_time
    # (Kode speech recognition kamu sudah oke)
    with microphone as source:
        print("Mendengarkan perintah 'dit tolongin dit'...")
        while True:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.2) # Sesuaikan jika perlu
                audio = recognizer.listen(source, phrase_time_limit=3, timeout=5) # Tambah timeout
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
                break # Keluar dari loop thread jika ada error fatal lain
speech_thread = threading.Thread(target=listen_for_command, daemon=True)
speech_thread.start()

# --------------- LOOP UTAMA OPENCV ---------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Tidak bisa membuka kamera.")
    exit()
# Coba set resolusi yang lebih umum jika default tidak bekerja baik
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
# Jika di atas terlalu berat, kembali ke 640x480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break
    frame_height, frame_width = frame.shape[:2]

    # Inisialisasi posisi awal Denis di tengah layar jika belum
    if not denis_initialized:
        denis_current_x = frame_width // 2
        denis_current_y = frame_height // 2
        denis_initialized = True

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Gambar labirin dan siapkan untuk collision detection
    current_display_frame = frame.copy() # Mulai dengan frame kamera asli
    if labirin_pil_original:
        # Resize labirin sekali per frame untuk display dan collision
        labirin_pil_resized_for_collision = labirin_pil_original.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        # Overlay labirin ke frame yang akan ditampilkan
        current_display_frame = overlay_image_alpha(current_display_frame, labirin_pil_resized_for_collision, 0, 0)
    else:
        labirin_pil_resized_for_collision = None # Pastikan ini None jika tidak ada labirin


    # Gambar hint jika aktif
    if show_hint:
        if time.time() - hint_display_start_time < HINT_DURATION:
            if hint_labirin_pil_original:
                hint_resized_pil = hint_labirin_pil_original.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
                current_display_frame = overlay_image_alpha(current_display_frame, hint_resized_pil, 0, 0)
        else:
            show_hint = False

    # Tentukan target posisi Denis dari hidung
    target_denis_x = denis_current_x # Default ke posisi sekarang
    target_denis_y = denis_current_y
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            target_denis_x = int(nose_tip.x * frame_width)
            target_denis_y = int(nose_tip.y * frame_height)

    # Pilih frame Denis yang akan ditampilkan (animasi GIF)
    denis_to_draw_pil = None
    if denis_frames_pil:
        current_frame_delay = gif_frame_durations[current_denis_frame_idx] if gif_frame_durations else DENIS_DEFAULT_FRAME_DELAY
        if time.time() - last_denis_frame_time > current_frame_delay:
            current_denis_frame_idx = (current_denis_frame_idx + 1) % len(denis_frames_pil)
            last_denis_frame_time = time.time()
        denis_to_draw_pil = denis_frames_pil[current_denis_frame_idx]

    # Hitung ukuran Denis untuk digambar dan collision
    denis_w_target, denis_h_target = 80, 80 # Default
    if denis_to_draw_pil:
        original_w, original_h = denis_to_draw_pil.size
        if original_w > 0: # Hindari division by zero
            denis_h_target = int((original_h / original_w) * denis_w_target)
        if denis_h_target == 0: denis_h_target = denis_w_target
        
        denis_resized_pil = denis_to_draw_pil.resize((denis_w_target, denis_h_target), Image.Resampling.LANCZOS)
    else: # Jika tidak ada frame Denis (seharusnya tidak terjadi jika fallback bekerja)
        denis_resized_pil = None


    # MODIFIKASI: Logika pergerakan Denis dengan deteksi tabrakan
    # Hitung arah dari posisi Denis saat ini ke target (hidung)
    delta_x = target_denis_x - (denis_current_x + denis_w_target // 2) # Target adalah pusat Denis
    delta_y = target_denis_y - (denis_current_y + denis_h_target // 2) # Target adalah pusat Denis (sedikit di atas hidung)

    distance = np.sqrt(delta_x**2 + delta_y**2)
    
    next_pos_x_candidate = denis_current_x
    next_pos_y_candidate = denis_current_y

    if distance > DENIS_SPEED: # Hanya bergerak jika jarak lebih besar dari kecepatan
        # Normalisasi vektor arah
        move_x_normalized = delta_x / distance
        move_y_normalized = delta_y / distance
        
        # Hitung langkah gerakan
        step_x = int(move_x_normalized * DENIS_SPEED)
        step_y = int(move_y_normalized * DENIS_SPEED)

        # Posisi berikutnya yang diinginkan (kiri atas bounding box Denis)
        next_pos_x_candidate = denis_current_x + step_x
        next_pos_y_candidate = denis_current_y + step_y
    elif distance > 1 : # Jika dekat, langsung ke target (snap kecil)
        next_pos_x_candidate = target_denis_x - denis_w_target // 2
        next_pos_y_candidate = target_denis_y - denis_h_target // 2 - 10 # -10 seperti offset gambar awal

    # Bounding box Denis di posisi berikutnya (kiri atas, kanan bawah)
    denis_bbox_next_x_min = next_pos_x_candidate + DENIS_COLLISION_PADDING
    denis_bbox_next_y_min = next_pos_y_candidate + DENIS_COLLISION_PADDING
    denis_bbox_next_x_max = next_pos_x_candidate + denis_w_target - DENIS_COLLISION_PADDING
    denis_bbox_next_y_max = next_pos_y_candidate + denis_h_target - DENIS_COLLISION_PADDING
    
    denis_bbox_next_tuple = (
        denis_bbox_next_x_min, denis_bbox_next_y_min,
        denis_bbox_next_x_max, denis_bbox_next_y_max
    )

    # Cek tabrakan untuk gerakan X
    temp_bbox_x = (denis_bbox_next_x_min, denis_current_y + DENIS_COLLISION_PADDING, denis_bbox_next_x_max, denis_current_y + denis_h_target - DENIS_COLLISION_PADDING)
    collision_x = check_collision(temp_bbox_x, labirin_pil_resized_for_collision)
    if not collision_x:
        denis_current_x = next_pos_x_candidate # Update posisi X jika tidak ada tabrakan
    
    # Cek tabrakan untuk gerakan Y (dari posisi X yang mungkin sudah diupdate atau belum)
    temp_bbox_y = (denis_current_x + DENIS_COLLISION_PADDING, denis_bbox_next_y_min, denis_current_x + denis_w_target - DENIS_COLLISION_PADDING, denis_bbox_next_y_max)
    collision_y = check_collision(temp_bbox_y, labirin_pil_resized_for_collision)
    if not collision_y:
        denis_current_y = next_pos_y_candidate # Update posisi Y jika tidak ada tabrakan

    # Gambar Denis di posisi (denis_current_x, denis_current_y) yang sudah divalidasi
    if denis_resized_pil:
        # Offset gambar agar hidung berada di sekitar tengah bawah Denis
        draw_x = denis_current_x
        draw_y = denis_current_y # Posisi kiri atas Denis
        current_display_frame = overlay_image_alpha(current_display_frame, denis_resized_pil, draw_x, draw_y)


    cv2.imshow('Dit Tolongin Dit Maze Filter', current_display_frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --------------- CLEANUP ---------------
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
if speech_thread.is_alive(): # Meskipun daemon, bisa coba join dengan timeout
    speech_thread.join(timeout=1.0)
print("Filter ditutup.")