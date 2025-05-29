import cv2
import mediapipe as mp
import speech_recognition as sr
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance # BARU: Import ImageEnhance
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

WALL_COLOR_THRESHOLD = 60
DENIS_SPEED = 5
DENIS_COLLISION_PADDING = 2

# Global untuk posisi Denis (akan diupdate dengan collision detection)
denis_current_x = 0 # Akan diinisialisasi oleh logika spawn
denis_current_y = 0 # Akan diinisialisasi oleh logika spawn

# BARU: Konfigurasi dan State untuk spawn dan aktivasi kontrol
denis_spawned_safely = False    # True jika Denis sudah ditempatkan di posisi awal
denis_controls_active = False # True jika hidung sudah sinkron dan kontrol aktif
ALIGNMENT_THRESHOLD = 60      # Jarak (pixel) hidung ke pusat Denis untuk aktivasi
INITIAL_SPAWN_Y_OFFSET = 5    # Sedikit offset dari atas untuk spawn awal Denis

# --------------- MEMUAT ASET GAMBAR ---------------
# ... (fungsi load_image_with_alpha tidak berubah) ...
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
    # Sebaiknya exit jika tidak ada aset Denis sama sekali
    # exit()

# --------------- FUNGSI OVERLAY & COLLISION ---------------
# ... (fungsi overlay_image_alpha tidak berubah) ...
# ... (fungsi check_collision tidak berubah) ...
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
    alpha_channel = overlay_roi[:, :, 3]
    if alpha_channel.dtype != np.float32 and alpha_channel.dtype != np.float64 :
        alpha = alpha_channel / 255.0
    else:
        alpha = alpha_channel
    if np.any(alpha < 1.0):
        alpha_expanded = np.expand_dims(alpha, axis=2)
        background_roi[:, :, :3] = (alpha_expanded * overlay_roi[:, :, :3] +
                                     (1 - alpha_expanded) * background_roi[:, :, :3])
    else:
        alpha_expanded = np.expand_dims(alpha, axis=2)
        background_roi[:, :, :3] = (alpha_expanded * overlay_roi[:, :, :3] +
                                     (1 - alpha_expanded) * background_roi[:, :, :3])
    return background_cv

def check_collision(denis_bbox_next, maze_image_pil):
    if maze_image_pil is None:
        return False
    x_min, y_min, x_max, y_max = denis_bbox_next
    check_points_relative = [
        (0, 0), (x_max - x_min, 0),
        (0, y_max - y_min), (x_max - x_min, y_max - y_min),
        ((x_max - x_min) // 2, 0),
        ((x_max - x_min) // 2, y_max - y_min),
        (0, (y_max - y_min) // 2),
        (x_max - x_min, (y_max - y_min) // 2)
    ]
    maze_w, maze_h = maze_image_pil.size
    maze_gray_pil = maze_image_pil.convert('L')
    for rel_x, rel_y in check_points_relative:
        abs_x = x_min + rel_x
        abs_y = y_min + rel_y
        if 0 <= abs_x < maze_w and 0 <= abs_y < maze_h:
            pixel_color = maze_gray_pil.getpixel((abs_x, abs_y))
            if pixel_color < WALL_COLOR_THRESHOLD:
                return True
    return False

# --------------- MEDIAPIPE FACE MESH ---------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --------------- SPEECH RECOGNITION ---------------
# ... (fungsi listen_for_command tidak berubah) ...
recognizer = sr.Recognizer()
microphone = sr.Microphone()
def listen_for_command():
    global show_hint, hint_display_start_time
    with microphone as source:
        print("Mendengarkan perintah 'dit tolongin dit'...")
        while True:
            try:
                recognizer.adjust_for_ambient_noise(source, duration=0.2)
                audio = recognizer.listen(source, phrase_time_limit=3, timeout=5)
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
        print("Gagal membaca frame dari kamera.")
        break
    frame_height, frame_width = frame.shape[:2]

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # Gambar labirin dan siapkan untuk collision detection
    current_display_frame = frame.copy()
    if labirin_pil_original:
        labirin_pil_resized_for_collision = labirin_pil_original.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        current_display_frame = overlay_image_alpha(current_display_frame, labirin_pil_resized_for_collision, 0, 0)
    else:
        labirin_pil_resized_for_collision = None

    # Gambar hint jika aktif
    if show_hint:
        if time.time() - hint_display_start_time < HINT_DURATION:
            if hint_labirin_pil_original:
                hint_resized_pil = hint_labirin_pil_original.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
                current_display_frame = overlay_image_alpha(current_display_frame, hint_resized_pil, 0, 0)
        else:
            show_hint = False

    # Tentukan target posisi Denis dari hidung
    # Default target ke tengah layar jika tidak ada wajah terdeteksi ATAU kontrol belum aktif
    # (agar Denis tidak lari ke (0,0) jika wajah hilang saat kontrol aktif)
    target_denis_x = frame_width // 2
    target_denis_y = frame_height // 2
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
    denis_w_target, denis_h_target = 80, 80
    if denis_to_draw_pil:
        original_w, original_h = denis_to_draw_pil.size
        if original_w > 0:
            denis_h_target = int((original_h / original_w) * denis_w_target)
        if denis_h_target == 0: denis_h_target = denis_w_target
        denis_resized_pil = denis_to_draw_pil.resize((denis_w_target, denis_h_target), Image.Resampling.LANCZOS)
    else:
        denis_resized_pil = None


    # BARU: Logika Spawn Awal Denis yang Aman & Aktivasi Kontrol
    if not denis_spawned_safely:
        # Coba spawn di pintu masuk atas (tengah-atas)
        spawn_center_x = frame_width // 2
        # spawn_y_top_edge = INITIAL_SPAWN_Y_OFFSET # Jarak dari atas layar ke tepi atas Denis
        spawn_y_top_edge = denis_h_target // 2 + INITIAL_SPAWN_Y_OFFSET # Lebih baik, agar bagian tengahnya sedikit di bawah

        denis_current_x = spawn_center_x - denis_w_target // 2
        denis_current_y = spawn_y_top_edge

        # Cek tabrakan di posisi spawn awal
        collided_at_spawn = False
        if labirin_pil_resized_for_collision:
            initial_denis_bbox = (
                denis_current_x + DENIS_COLLISION_PADDING,
                denis_current_y + DENIS_COLLISION_PADDING,
                denis_current_x + denis_w_target - DENIS_COLLISION_PADDING,
                denis_current_y + denis_h_target - DENIS_COLLISION_PADDING
            )
            if check_collision(initial_denis_bbox, labirin_pil_resized_for_collision):
                collided_at_spawn = True
                print("WARNING: Posisi spawn awal (atas-tengah) bertabrakan.")

        if collided_at_spawn or not labirin_pil_resized_for_collision: # Jika tabrakan atau tidak ada labirin
             # Fallback ke tengah layar
            print("Mencoba spawn di tengah layar sebagai fallback.")
            denis_current_x = frame_width // 2 - denis_w_target // 2
            denis_current_y = frame_height // 2 - denis_h_target // 2
            # Opsional: cek lagi tabrakan di tengah, tapi untuk sekarang kita terima saja
            if labirin_pil_resized_for_collision: # Cek lagi jika fallback ke tengah
                fallback_denis_bbox = (
                    denis_current_x + DENIS_COLLISION_PADDING,
                    denis_current_y + DENIS_COLLISION_PADDING,
                    denis_current_x + denis_w_target - DENIS_COLLISION_PADDING,
                    denis_current_y + denis_h_target - DENIS_COLLISION_PADDING
                )
                if check_collision(fallback_denis_bbox, labirin_pil_resized_for_collision):
                    print("WARNING: Posisi spawn fallback (tengah layar) juga bertabrakan! Denis mungkin terjebak.")

        denis_spawned_safely = True
        denis_controls_active = False # Pastikan kontrol belum aktif
        print(f"Denis spawned at: ({denis_current_x}, {denis_current_y})")

    # BARU: Cek sinkronisasi hidung untuk aktivasi kontrol
    if denis_spawned_safely and not denis_controls_active:
        if results.multi_face_landmarks: # Hanya cek jika wajah terdeteksi
            denis_center_x_current = denis_current_x + denis_w_target // 2
            denis_center_y_current = denis_current_y + denis_h_target // 2

            # target_denis_x dan target_denis_y sudah dihitung dari hidung di atas
            dist_to_align = np.sqrt((target_denis_x - denis_center_x_current)**2 + (target_denis_y - denis_center_y_current)**2)

            if dist_to_align < ALIGNMENT_THRESHOLD:
                denis_controls_active = True
                print("Kontrol diaktifkan!")
            else:
                # Tampilkan instruksi jika belum sinkron
                cv2.putText(current_display_frame, "Arahkan hidung ke karakter untuk mulai",
                            (frame_width // 2 - 250, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (200, 200, 255), 2)


    # Logika pergerakan Denis dengan deteksi tabrakan (HANYA JIKA KONTROL AKTIF)
    if denis_spawned_safely and denis_controls_active:
        delta_x = target_denis_x - (denis_current_x + denis_w_target // 2)
        delta_y = target_denis_y - (denis_current_y + denis_h_target // 2)
        distance = np.sqrt(delta_x**2 + delta_y**2)
        next_pos_x_candidate = denis_current_x
        next_pos_y_candidate = denis_current_y

        if distance > DENIS_SPEED:
            move_x_normalized = delta_x / distance
            move_y_normalized = delta_y / distance
            
            # Perbaikan untuk integer truncation, coba gerak minimal 1 px jika ada delta
            raw_step_x = move_x_normalized * DENIS_SPEED
            raw_step_y = move_y_normalized * DENIS_SPEED

            if abs(raw_step_x) < 1.0 and abs(raw_step_x) > 0.001 : # Jika float step < 1 tapi > 0
                step_x = int(np.sign(raw_step_x))
            else:
                step_x = int(raw_step_x)

            if abs(raw_step_y) < 1.0 and abs(raw_step_y) > 0.001 :
                step_y = int(np.sign(raw_step_y))
            else:
                step_y = int(raw_step_y)

            next_pos_x_candidate = denis_current_x + step_x
            next_pos_y_candidate = denis_current_y + step_y
        elif distance > 1 : # Snap kecil
            next_pos_x_candidate = target_denis_x - denis_w_target // 2
            next_pos_y_candidate = target_denis_y - denis_h_target // 2 - 10 # Offset agar hidung di tengah-bawah Denis

        denis_bbox_next_x_min = next_pos_x_candidate + DENIS_COLLISION_PADDING
        denis_bbox_next_y_min = next_pos_y_candidate + DENIS_COLLISION_PADDING
        denis_bbox_next_x_max = next_pos_x_candidate + denis_w_target - DENIS_COLLISION_PADDING
        denis_bbox_next_y_max = next_pos_y_candidate + denis_h_target - DENIS_COLLISION_PADDING

        temp_bbox_x = (denis_bbox_next_x_min, denis_current_y + DENIS_COLLISION_PADDING, denis_bbox_next_x_max, denis_current_y + denis_h_target - DENIS_COLLISION_PADDING)
        collision_x = check_collision(temp_bbox_x, labirin_pil_resized_for_collision)
        if not collision_x:
            denis_current_x = next_pos_x_candidate
        
        temp_bbox_y = (denis_current_x + DENIS_COLLISION_PADDING, denis_bbox_next_y_min, denis_current_x + denis_w_target - DENIS_COLLISION_PADDING, denis_bbox_next_y_max)
        collision_y = check_collision(temp_bbox_y, labirin_pil_resized_for_collision)
        if not collision_y:
            denis_current_y = next_pos_y_candidate

    # Gambar Denis di posisi (denis_current_x, denis_current_y)
    denis_to_overlay_pil = denis_resized_pil # Defaultnya adalah frame Denis yang sudah di-resize
    if denis_resized_pil:
        # BARU: Redupkan Denis jika kontrol belum aktif
        if denis_spawned_safely and not denis_controls_active:
            try:
                # Buat salinan untuk dimodifikasi kecerahannya
                temp_denis_pil = denis_resized_pil.copy()
                enhancer = ImageEnhance.Brightness(temp_denis_pil)
                denis_to_overlay_pil = enhancer.enhance(0.5) # Redupkan jadi 50%
            except Exception as e:
                # print(f"Error saat meredupkan Denis: {e}")
                denis_to_overlay_pil = denis_resized_pil # Fallback ke gambar asli jika error
        
        draw_x = denis_current_x
        draw_y = denis_current_y
        current_display_frame = overlay_image_alpha(current_display_frame, denis_to_overlay_pil, draw_x, draw_y)


    cv2.imshow('Dit Tolongin Dit Maze Filter', current_display_frame)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# --------------- CLEANUP ---------------
face_mesh.close()
cap.release()
cv2.destroyAllWindows()
if speech_thread.is_alive():
    speech_thread.join(timeout=1.0)
print("Filter ditutup.")