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

# Global variables
show_hint = False
hint_display_start_time = 0
HINT_DURATION = 5  # Detik
USE_GIF_DENIS = True

# Konfigurasi untuk deteksi tabrakan
WALL_COLOR_THRESHOLD = 60 # Anggap warna di bawah ini (pada skala abu-abu) adalah dinding
DENIS_SPEED = 5          # Piksel per frame, sesuaikan untuk kecepatan yang pas
DENIS_COLLISION_PADDING = 2 # Sedikit padding untuk deteksi agar tidak terlalu mepet

# Global untuk posisi Denis (pojok kiri atas)
denis_current_x = 0
denis_current_y = 0
denis_initialized = False # Flag untuk menandai apakah posisi awal Denis sudah diset

# Ukuran default Denis, akan diupdate jika GIF/PNG dimuat
denis_w_target, denis_h_target = 80, 80

# PENTING: Tentukan koordinat START di labirin ORIGINAL (sebelum di-resize)
# GANTI NILAI INI SESUAI GAMBAR LABIRINMU!
# Misal, jika labirin originalmu 1000x1000 px, dan start ada di (50, 50)
LABIRIN_START_X_ORIGINAL = 50  # Koordinat X titik start di labirin.png (pojok kiri atas area start)
LABIRIN_START_Y_ORIGINAL = 50  # Koordinat Y titik start di labirin.png (pojok kiri atas area start)

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
labirin_pil_original = load_image_with_alpha(LABIRIN_PATH)
hint_labirin_pil_original = load_image_with_alpha(HINT_LABIRIN_PATH)

labirin_pil_resized_for_collision = None # Akan diupdate per frame

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
            print("Warning: Gagal memuat frame dari GIF. Cek file GIF.")
            USE_GIF_DENIS = False
        else:
            print(f"Berhasil memuat {len(denis_frames_pil)} frame dari GIF.")
            # Update ukuran Denis berdasarkan frame pertama GIF jika ada
            if denis_frames_pil:
                first_frame_w, first_frame_h = denis_frames_pil[0].size
                aspect_ratio = first_frame_h / first_frame_w if first_frame_w > 0 else 1
                denis_w_target = 80 # Tetapkan lebar target
                denis_h_target = int(denis_w_target * aspect_ratio)
                if denis_h_target == 0: denis_h_target = denis_w_target

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
    # Update ukuran Denis berdasarkan gambar diam
    diam_w, diam_h = denis_diam_pil.size
    aspect_ratio = diam_h / diam_w if diam_w > 0 else 1
    denis_w_target = 80 # Tetapkan lebar target
    denis_h_target = int(denis_w_target * aspect_ratio)
    if denis_h_target == 0: denis_h_target = denis_w_target

elif not denis_frames_pil and not denis_diam_pil:
     print("CRITICAL: Tidak ada aset Denis (GIF atau diam) yang bisa dimuat.")

# --------------- FUNGSI OVERLAY & COLLISION ---------------
def overlay_image_alpha(background_cv, overlay_pil, x, y):
    if overlay_pil is None:
        return background_cv
    
    # Pastikan x dan y adalah integer
    x, y = int(x), int(y)

    overlay_cv = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGBA2BGRA)
    h, w = overlay_cv.shape[:2]
    bh, bw = background_cv.shape[:2]

    # Tentukan ROI pada background dan overlay
    x_start_bg, y_start_bg = x, y
    x_end_bg, y_end_bg = x + w, y + h

    x_start_overlay, y_start_overlay = 0, 0
    x_end_overlay, y_end_overlay = w, h

    # Cek dan sesuaikan jika overlay keluar batas background
    if x_start_bg < 0:
        x_start_overlay = -x_start_bg
        x_start_bg = 0
    if y_start_bg < 0:
        y_start_overlay = -y_start_bg
        y_start_bg = 0
    if x_end_bg > bw:
        x_end_overlay -= (x_end_bg - bw)
        x_end_bg = bw
    if y_end_bg > bh:
        y_end_overlay -= (y_end_bg - bh)
        y_end_bg = bh
    
    # Jika setelah penyesuaian, lebar atau tinggi overlay menjadi 0 atau negatif, jangan lakukan apa-apa
    overlay_roi_w = x_end_overlay - x_start_overlay
    overlay_roi_h = y_end_overlay - y_start_overlay
    if overlay_roi_w <= 0 or overlay_roi_h <= 0:
        return background_cv

    # Ambil ROI
    background_roi = background_cv[y_start_bg:y_end_bg, x_start_bg:x_end_bg]
    overlay_roi_full = overlay_cv[y_start_overlay:y_end_overlay, x_start_overlay:x_end_overlay]
    
    # Pastikan ROI tidak kosong (mungkin terjadi jika overlay sepenuhnya di luar frame)
    if background_roi.size == 0 or overlay_roi_full.size == 0:
        return background_cv
    
    # Alpha blending
    alpha_channel = overlay_roi_full[:, :, 3] / 255.0
    alpha_expanded = np.expand_dims(alpha_channel, axis=2) # Untuk broadcasting (H, W, 1)

    try:
        blended_roi = (alpha_expanded * overlay_roi_full[:, :, :3] +
                       (1 - alpha_expanded) * background_roi[:, :, :3])
        background_cv[y_start_bg:y_end_bg, x_start_bg:x_end_bg] = blended_roi.astype(np.uint8)
    except ValueError as e:
        # Sering terjadi jika ROI tidak pas karena pembulatan float ke int atau overlay di tepi.
        # print(f"ValueError during alpha blending: {e}")
        # print(f"Background ROI shape: {background_roi.shape}, Overlay ROI shape: {overlay_roi_full.shape}, Alpha shape: {alpha_expanded.shape}")
        # Coba pastikan shapes cocok jika memungkinkan, atau lewati blending untuk frame ini
        pass
    return background_cv

def check_collision(denis_bbox_next, maze_image_pil):
    if maze_image_pil is None:
        return False # Tidak ada labirin, tidak ada tabrakan

    x_min, y_min, x_max, y_max = map(int, denis_bbox_next) # Pastikan integer

    # Jika bounding box tidak valid (misal, min > max)
    if x_min >= x_max or y_min >= y_max:
        return False

    # Titik-titik sampel di tepi bounding box Denis (relatif terhadap bbox)
    # Mengurangi 1 agar tidak keluar batas saat bbox lebarnya hanya 1 piksel
    width_bbox = max(1, x_max - x_min)
    height_bbox = max(1, y_max - y_min)

    check_points_relative = [
        (0, 0), (width_bbox - 1, 0),  # Kiri atas, Kanan atas
        (0, height_bbox - 1), (width_bbox - 1, height_bbox - 1), # Kiri bawah, Kanan bawah
        (width_bbox // 2, 0), # Tengah atas
        (width_bbox // 2, height_bbox - 1), # Tengah bawah
        (0, height_bbox // 2), # Tengah kiri
        (width_bbox - 1, height_bbox // 2) # Tengah kanan
    ]

    maze_w, maze_h = maze_image_pil.size
    try:
        maze_gray_pil = maze_image_pil.convert('L') # Konversi ke grayscale
    except Exception as e:
        # print(f"Error converting maze to grayscale: {e}")
        return False # Anggap tidak bisa cek collision

    for rel_x, rel_y in check_points_relative:
        abs_x = x_min + rel_x
        abs_y = y_min + rel_y

        # Pastikan titik berada dalam batas gambar labirin
        if 0 <= abs_x < maze_w and 0 <= abs_y < maze_h:
            try:
                pixel_color = maze_gray_pil.getpixel((abs_x, abs_y))
                if pixel_color < WALL_COLOR_THRESHOLD:
                    # print(f"Collision at ({abs_x}, {abs_y}), color: {pixel_color}")
                    return True  # Tabrakan terdeteksi
            except IndexError: # Seharusnya sudah dicegah oleh `0 <= abs_x < maze_w` dst.
                # print(f"IndexError at getpixel: ({abs_x}, {abs_y}) vs maze ({maze_w}x{maze_h})")
                return True # Anggap tabrakan jika keluar batas
            except Exception as e:
                # print(f"Error getting pixel for collision check: {e}")
                return True # Anggap tabrakan jika ada error lain
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
    with microphone as source:
        print("Mendengarkan perintah 'dit tolongin dit'...")
        while True:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = recognizer.listen(source, phrase_time_limit=3, timeout=5) # Timeout untuk listen
                command = recognizer.recognize_google(audio, language="id-ID").lower()
                print(f"Diterima: {command}")
                if "dit tolongin dit" in command:
                    print("Hint diminta!")
                    show_hint = True
                    hint_display_start_time = time.time()
            except sr.WaitTimeoutError:
                pass # Tidak ada suara dalam timeout, loop lagi
            except sr.UnknownValueError:
                pass # Tidak bisa mengenali ucapan
            except sr.RequestError as e:
                print(f"Error dari layanan Google Speech Recognition; {e}")
                time.sleep(2) # Beri jeda sebelum mencoba lagi
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
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Gagal membaca frame dari kamera.")
        break
    frame_height, frame_width = frame.shape[:2]

    if not denis_initialized:
        if labirin_pil_original:
            original_maze_w, original_maze_h = labirin_pil_original.size
            if original_maze_w > 0 and original_maze_h > 0: # Pastikan ukuran valid
                scale_x = frame_width / original_maze_w
                scale_y = frame_height / original_maze_h

                start_pos_center_x_on_frame = int(LABIRIN_START_X_ORIGINAL * scale_x + (denis_w_target * scale_x / 2)) # Titik tengah area start
                start_pos_center_y_on_frame = int(LABIRIN_START_Y_ORIGINAL * scale_y + (denis_h_target * scale_y / 2)) # Titik tengah area start

                denis_current_x = start_pos_center_x_on_frame - (denis_w_target // 2)
                denis_current_y = start_pos_center_y_on_frame - (denis_h_target // 2)

                denis_current_x = max(0, min(denis_current_x, frame_width - denis_w_target))
                denis_current_y = max(0, min(denis_current_y, frame_height - denis_h_target))
                print(f"Denis initialized at ({denis_current_x}, {denis_current_y}) based on maze start.")
            else: # Fallback jika ukuran labirin original tidak valid
                denis_current_x = (frame_width - denis_w_target) // 2
                denis_current_y = (frame_height - denis_h_target) // 2
        else:
            denis_current_x = (frame_width - denis_w_target) // 2
            denis_current_y = (frame_height - denis_h_target) // 2
            print("Denis initialized at center (no maze for start point).")
        denis_initialized = True

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Mulai dengan frame kamera, lalu overlay elemen lain
    current_display_frame = frame.copy()

    if labirin_pil_original:
        labirin_pil_resized_for_collision = labirin_pil_original.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        current_display_frame = overlay_image_alpha(current_display_frame, labirin_pil_resized_for_collision, 0, 0)
    else:
        labirin_pil_resized_for_collision = None

    if show_hint:
        if time.time() - hint_display_start_time < HINT_DURATION:
            if hint_labirin_pil_original:
                hint_resized_pil = hint_labirin_pil_original.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
                current_display_frame = overlay_image_alpha(current_display_frame, hint_resized_pil, 0, 0)
        else:
            show_hint = False

    # Target posisi (pusat Denis) dari hidung
    # Default ke posisi tengah Denis saat ini jika wajah tidak terdeteksi
    target_denis_center_x = denis_current_x + denis_w_target // 2
    target_denis_center_y = denis_current_y + denis_h_target // 2
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1] # Landmark hidung
            target_denis_center_x = int(nose_tip.x * frame_width)
            # Offset Y agar Denis sedikit di atas hidung, atau sesuai preferensi
            target_denis_center_y = int(nose_tip.y * frame_height) - (denis_h_target // 2) # Bagian bawah Denis sejajar hidung

    # Animasi Denis
    denis_to_draw_pil = None
    if denis_frames_pil:
        current_frame_delay = gif_frame_durations[current_denis_frame_idx % len(gif_frame_durations)] if gif_frame_durations else DENIS_DEFAULT_FRAME_DELAY
        if time.time() - last_denis_frame_time > current_frame_delay:
            current_denis_frame_idx = (current_denis_frame_idx + 1) % len(denis_frames_pil)
            last_denis_frame_time = time.time()
        denis_to_draw_pil = denis_frames_pil[current_denis_frame_idx]

    denis_resized_pil = None
    if denis_to_draw_pil:
        # Gunakan denis_w_target dan denis_h_target yang sudah dihitung saat load aset
        if denis_w_target > 0 and denis_h_target > 0:
             denis_resized_pil = denis_to_draw_pil.resize((denis_w_target, denis_h_target), Image.Resampling.LANCZOS)

    # --- Logika Pergerakan Denis dengan Deteksi Tabrakan ---
    # Posisi pusat Denis saat ini
    current_denis_center_x = denis_current_x + denis_w_target // 2
    current_denis_center_y = denis_current_y + denis_h_target // 2

    delta_x = target_denis_center_x - current_denis_center_x
    delta_y = target_denis_center_y - current_denis_center_y
    distance = np.sqrt(delta_x**2 + delta_y**2)
    
    # Calon posisi kiri atas Denis berikutnya
    next_pos_x_candidate_topleft = denis_current_x
    next_pos_y_candidate_topleft = denis_current_y

    if distance > DENIS_SPEED: # Hanya bergerak jika jarak signifikan
        move_x_normalized = delta_x / distance
        move_y_normalized = delta_y / distance
        step_x = int(move_x_normalized * DENIS_SPEED)
        step_y = int(move_y_normalized * DENIS_SPEED)
        next_pos_x_candidate_topleft = denis_current_x + step_x
        next_pos_y_candidate_topleft = denis_current_y + step_y
    elif distance > 1: # Jika sudah dekat, snap ke target
        next_pos_x_candidate_topleft = target_denis_center_x - (denis_w_target // 2)
        next_pos_y_candidate_topleft = target_denis_center_y - (denis_h_target // 2)

    # Bounding box Denis di posisi berikutnya (kiri atas, kanan bawah) untuk collision check
    # Padding sudah termasuk dalam logika check_collision jika diperlukan, atau bisa ditambahkan di sini
    bbox_x_min = next_pos_x_candidate_topleft + DENIS_COLLISION_PADDING
    bbox_y_min = next_pos_y_candidate_topleft + DENIS_COLLISION_PADDING
    bbox_x_max = next_pos_x_candidate_topleft + denis_w_target - DENIS_COLLISION_PADDING
    bbox_y_max = next_pos_y_candidate_topleft + denis_h_target - DENIS_COLLISION_PADDING
    
    # Cek Tabrakan X
    # Bounding box untuk cek X: X baru, Y lama (posisi kiri atas saat ini)
    temp_bbox_x = (bbox_x_min, denis_current_y + DENIS_COLLISION_PADDING, bbox_x_max, denis_current_y + denis_h_target - DENIS_COLLISION_PADDING)
    collision_x = check_collision(temp_bbox_x, labirin_pil_resized_for_collision)
    if not collision_x:
        denis_current_x = next_pos_x_candidate_topleft
    
    # Cek Tabrakan Y
    # Bounding box untuk cek Y: X (mungkin sudah update), Y baru
    temp_bbox_y = (denis_current_x + DENIS_COLLISION_PADDING, bbox_y_min, denis_current_x + denis_w_target - DENIS_COLLISION_PADDING, bbox_y_max)
    collision_y = check_collision(temp_bbox_y, labirin_pil_resized_for_collision)
    if not collision_y:
        denis_current_y = next_pos_y_candidate_topleft

    # Gambar Denis di posisi (denis_current_x, denis_current_y) yang sudah divalidasi
    if denis_resized_pil:
        current_display_frame = overlay_image_alpha(current_display_frame, denis_resized_pil, denis_current_x, denis_current_y)

    cv2.imshow('Dit Tolongin Dit Maze Filter', current_display_frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --------------- CLEANUP ---------------
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
if speech_thread.is_alive():
    speech_thread.join(timeout=1.0) # Coba join thread dengan timeout
print("Filter ditutup.")