import cv2
import mediapipe as mp
import speech_recognition as sr
import numpy as np
from PIL import Image, ImageSequence, ImageEnhance
import threading
import time
import os

# --------------- KONFIGURASI & SETUP AWAL ---------------
ASSETS_DIR = "assets"
DENIS_DIAM_PATH = os.path.join(ASSETS_DIR, "denis_diam.png")
LABIRIN_PATH = os.path.join(ASSETS_DIR, "labirin.png") # Pastikan ini hitam-putih solid
HINT_LABIRIN_PATH = os.path.join(ASSETS_DIR, "hint_labirin.png")
DENIS_GIF_PATH = os.path.join(ASSETS_DIR, "denis_bergerak.gif")

show_hint = False
hint_display_start_time = 0
HINT_DURATION = 5
USE_GIF_DENIS = True

# PENYESUAIAN PARAMETER KRITIS:
WALL_COLOR_THRESHOLD = 120 # Dinaikkan (asumsi jalur putih >120, dinding <120 setelah grayscale)
DENIS_SPEED = 3           # Kecepatan mungkin perlu disesuaikan dengan hitbox baru

HITBOX_RADIUS = 3         # Radius hitbox lingkaran Denis (kecil agar muat di jalur)

denis_current_x = 0       # Pusat hitbox X
denis_current_y = 0       # Pusat hitbox Y

denis_spawned_safely = False
denis_controls_active = False
ALIGNMENT_THRESHOLD = 60      # Kurangi sedikit jika hitbox lebih kecil
INITIAL_SPAWN_Y_OFFSET = 10   # Jarak dari atas layar ke tepi ATAS hitbox saat spawn ideal

VISUAL_DENIS_WIDTH = 40       # Ukuran visual Denis (bisa disesuaikan)
# VISUAL_DENIS_HEIGHT akan dihitung dari aspek rasio

# --------------- MEMUAT ASET GAMBAR (Sama seperti sebelumnya) ---------------
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
            except: gif_frame_durations.append(DENIS_DEFAULT_FRAME_DELAY)
        if not denis_frames_pil: USE_GIF_DENIS = False; print("Gagal muat frame GIF.")
        else: print(f"Berhasil memuat {len(denis_frames_pil)} frame dari GIF.")
    except Exception as e: USE_GIF_DENIS = False; print(f"Error muat GIF: {e}")
if not denis_frames_pil and denis_diam_pil: # Fallback ke gambar diam
    print("Menggunakan Denis diam sebagai fallback animasi.")
    denis_frames_pil = [denis_diam_pil]; gif_frame_durations = [DENIS_DEFAULT_FRAME_DELAY]
elif not denis_frames_pil: print("KRITIS: Tidak ada aset Denis yang bisa dimuat.")


# --------------- FUNGSI OVERLAY & COLLISION (Sama seperti versi hitbox lingkaran) ---------------
def overlay_image_alpha(background_cv, overlay_pil, x, y):
    if overlay_pil is None: return background_cv
    overlay_cv = cv2.cvtColor(np.array(overlay_pil), cv2.COLOR_RGBA2BGRA)
    h, w = overlay_cv.shape[:2]
    bh, bw = background_cv.shape[:2]
    x_start, y_start = int(x), int(y)
    x_end, y_end = int(x + w), int(y + h)
    overlay_x_start, overlay_y_start = 0, 0
    if x_start < 0: overlay_x_start = -x_start; x_start = 0
    if y_start < 0: overlay_y_start = -y_start; y_start = 0
    if x_end > bw: x_end = bw
    if y_end > bh: y_end = bh
    overlay_w = x_end - x_start
    overlay_h = y_end - y_start
    if overlay_w <= 0 or overlay_h <= 0: return background_cv
    background_roi = background_cv[y_start:y_end, x_start:x_end]
    overlay_roi = overlay_cv[overlay_y_start : overlay_y_start + overlay_h, overlay_x_start : overlay_x_start + overlay_w]
    if background_roi.shape[0] != overlay_roi.shape[0] or background_roi.shape[1] != overlay_roi.shape[1]: return background_cv
    alpha_channel = overlay_roi[:, :, 3]
    alpha = alpha_channel / 255.0 if alpha_channel.dtype != np.float32 and alpha_channel.dtype != np.float64 else alpha_channel
    alpha_expanded = np.expand_dims(alpha, axis=2)
    blended_roi = (alpha_expanded * overlay_roi[:, :, :3] + (1 - alpha_expanded) * background_roi[:, :, :3])
    background_cv[y_start:y_end, x_start:x_end] = blended_roi.astype(background_cv.dtype)
    return background_cv

def check_collision_circle(cx, cy, radius, maze_image_pil):
    if maze_image_pil is None: return False
    maze_w, maze_h = maze_image_pil.size
    maze_gray_pil = maze_image_pil.convert('L')
    num_points_circumference = 8
    pcx, pcy = int(cx), int(cy)
    if not (0 <= pcx < maze_w and 0 <= pcy < maze_h): return True # Pusat di luar batas
    if maze_gray_pil.getpixel((pcx, pcy)) < WALL_COLOR_THRESHOLD: return True

    if radius > 0:
        for i in range(num_points_circumference):
            angle = 2 * np.pi * i / num_points_circumference
            check_x = int(cx + radius * np.cos(angle))
            check_y = int(cy + radius * np.sin(angle))
            if not (0 <= check_x < maze_w and 0 <= check_y < maze_h): return True # Titik di lingkaran keluar batas
            if maze_gray_pil.getpixel((check_x, check_y)) < WALL_COLOR_THRESHOLD: return True
    return False

# --------------- MEDIAPIPE & SPEECH RECOGNITION (Sama) ---------------
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
recognizer = sr.Recognizer()
microphone = sr.Microphone()
def listen_for_command():
    global show_hint, hint_display_start_time
    with microphone as source:
        print("Mendengarkan perintah 'dit tolongin dit'...")
        while True:
            try:
                audio = recognizer.listen(source, phrase_time_limit=3, timeout=None)
                command = recognizer.recognize_google(audio, language="id-ID").lower()
                print(f"Diterima: {command}")
                if "dit tolongin dit" in command or "tolongin dit" in command:
                    print("Hint diminta!"); show_hint = True; hint_display_start_time = time.time()
            except sr.WaitTimeoutError: pass
            except sr.UnknownValueError: pass
            except sr.RequestError as e: print(f"Error Google Speech: {e}"); time.sleep(2)
            except Exception as e: print(f"Error speech: {e}"); break
speech_thread = threading.Thread(target=listen_for_command, daemon=True); speech_thread.start()

# --------------- LOOP UTAMA OPENCV ---------------
cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Error: Kamera tidak bisa dibuka."); exit()
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
print("Tekan 'q' untuk keluar.")

while cap.isOpened():
    success, frame = cap.read()
    if not success: print("Gagal membaca frame."); break
    frame_height, frame_width = frame.shape[:2]
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    current_display_frame = frame.copy()
    if labirin_pil_original:
        labirin_pil_resized_for_collision = labirin_pil_original.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
        # DEBUG: Simpan grayscale maze sekali untuk diinspeksi manual jika perlu
        # if not denis_spawned_safely and labirin_pil_resized_for_collision:
        #     debug_gray_maze = labirin_pil_resized_for_collision.convert('L')
        #     debug_gray_maze.save("debug_grayscale_maze_live.png")
        #     print("Debug Grayscale Maze Disimpan ke debug_grayscale_maze_live.png")
        current_display_frame = overlay_image_alpha(current_display_frame, labirin_pil_resized_for_collision, 0, 0)
    else:
        labirin_pil_resized_for_collision = None

    if show_hint: # Logika Hint (Sama)
        if time.time() - hint_display_start_time < HINT_DURATION:
            if hint_labirin_pil_original:
                hint_resized_pil = hint_labirin_pil_original.resize((frame_width, frame_height), Image.Resampling.LANCZOS)
                current_display_frame = overlay_image_alpha(current_display_frame, hint_resized_pil, 0, 0)
        else: show_hint = False

    target_denis_x_nose = frame_width // 2 # Default jika wajah tidak terdeteksi
    target_denis_y_nose = frame_height // 2
    face_detected = results.multi_face_landmarks is not None
    if face_detected:
        for face_landmarks in results.multi_face_landmarks:
            nose_tip = face_landmarks.landmark[1]
            target_denis_x_nose = int(nose_tip.x * frame_width)
            target_denis_y_nose = int(nose_tip.y * frame_height)

    denis_to_draw_pil = None # Logika Animasi GIF (Sama)
    if denis_frames_pil:
        current_frame_delay = gif_frame_durations[current_denis_frame_idx] if gif_frame_durations else DENIS_DEFAULT_FRAME_DELAY
        if time.time() - last_denis_frame_time > current_frame_delay:
            current_denis_frame_idx = (current_denis_frame_idx + 1) % len(denis_frames_pil)
            last_denis_frame_time = time.time()
        denis_to_draw_pil = denis_frames_pil[current_denis_frame_idx]

    visual_denis_w, visual_denis_h = VISUAL_DENIS_WIDTH, VISUAL_DENIS_WIDTH # Ukuran Visual Denis
    if denis_to_draw_pil:
        original_w, original_h = denis_to_draw_pil.size
        if original_w > 0: visual_denis_h = int((original_h / original_w) * VISUAL_DENIS_WIDTH)
        if visual_denis_h == 0: visual_denis_h = VISUAL_DENIS_WIDTH
        denis_resized_pil_visual = denis_to_draw_pil.resize((VISUAL_DENIS_WIDTH, visual_denis_h), Image.Resampling.LANCZOS)
    else: denis_resized_pil_visual = None

    # Logika Spawn Awal Denis (PUSAT HITBOX) yang Aman (Sama seperti versi hitbox lingkaran)
    if not denis_spawned_safely:
        found_safe_spawn_point = False; spawn_attempts_log = []
        attempt_cx_ideal = frame_width // 2
        attempt_cy_ideal = INITIAL_SPAWN_Y_OFFSET + HITBOX_RADIUS
        is_colliding = True
        if labirin_pil_resized_for_collision: is_colliding = check_collision_circle(attempt_cx_ideal, attempt_cy_ideal, HITBOX_RADIUS, labirin_pil_resized_for_collision)
        else: is_colliding = False
        spawn_attempts_log.append(f"  Upaya 1 (Ideal Atas Ctr): ({attempt_cx_ideal},{attempt_cy_ideal}), R:{HITBOX_RADIUS}, Col: {is_colliding}")
        if not is_colliding: denis_current_x, denis_current_y = attempt_cx_ideal, attempt_cy_ideal; found_safe_spawn_point = True

        if not found_safe_spawn_point: # Upaya 2: Tengah
            attempt_cx_mid, attempt_cy_mid = frame_width // 2, frame_height // 2
            is_colliding = True
            if labirin_pil_resized_for_collision: is_colliding = check_collision_circle(attempt_cx_mid, attempt_cy_mid, HITBOX_RADIUS, labirin_pil_resized_for_collision)
            else: is_colliding = False
            spawn_attempts_log.append(f"  Upaya 2 (Tengah Ctr): ({attempt_cx_mid},{attempt_cy_mid}), R:{HITBOX_RADIUS}, Col: {is_colliding}")
            if not is_colliding: denis_current_x, denis_current_y = attempt_cx_mid, attempt_cy_mid; found_safe_spawn_point = True
        
        if not found_safe_spawn_point and labirin_pil_resized_for_collision: # Upaya 3: Pencarian
            search_origin_cx, search_origin_cy = frame_width // 2, frame_height // 2
            search_step = max(10, HITBOX_RADIUS * 4); max_search_rings = 5 # Sesuaikan search_step
            for ring in range(max_search_rings + 1):
                offsets = []
                if ring == 0:
                    if is_colliding: offsets.append((0,0)) # is_colliding dari upaya tengah layar
                else:
                    for i_s in range(-ring, ring + 1):
                        offsets.extend([(i_s, -ring), (i_s, ring), (-ring, i_s), (ring, i_s)])
                    offsets = list(set(offsets))
                for dx_mult, dy_mult in offsets:
                    prop_cx = search_origin_cx + dx_mult * search_step
                    prop_cy = search_origin_cy + dy_mult * search_step
                    prop_cx = max(HITBOX_RADIUS, min(prop_cx, frame_width - HITBOX_RADIUS -1))
                    prop_cy = max(HITBOX_RADIUS, min(prop_cy, frame_height - HITBOX_RADIUS -1))
                    is_coll_search = check_collision_circle(prop_cx, prop_cy, HITBOX_RADIUS, labirin_pil_resized_for_collision)
                    spawn_attempts_log.append(f"    Search R{ring}({dx_mult},{dy_mult}):Ctr({prop_cx},{prop_cy}),R:{HITBOX_RADIUS},Col:{is_coll_search}")
                    if not is_coll_search: denis_current_x,denis_current_y=prop_cx,prop_cy; found_safe_spawn_point=True; break
                if found_safe_spawn_point: break
            if not found_safe_spawn_point: spawn_attempts_log.append("  Pencarian iteratif gagal.")

        if not found_safe_spawn_point: # Fallback Akhir
            denis_current_x, denis_current_y = frame_width//2, frame_height//2
            spawn_attempts_log.append(f"  FALLBACK FINAL (Tengah Ctr):({denis_current_x},{denis_current_y})")
            print("KRITIS: Semua upaya spawn gagal. Denis di tengah layar (pusat hitbox).")
        print("Log Upaya Spawn:\n" + "\n".join(spawn_attempts_log))
        print(f"Denis (Pusat Hitbox) Akhirnya Spawn di: ({denis_current_x}, {denis_current_y})")
        denis_spawned_safely = True; denis_controls_active = False

    # Tentukan target efektif (jika wajah hilang setelah kontrol aktif, Denis diam)
    effective_target_x = target_denis_x_nose
    effective_target_y = target_denis_y_nose
    if denis_controls_active and not face_detected: # Wajah hilang saat kontrol aktif
        effective_target_x = denis_current_x # Targetkan posisi saat ini agar diam
        effective_target_y = denis_current_y

    # Cek sinkronisasi hidung untuk aktivasi kontrol
    if denis_spawned_safely and not denis_controls_active:
        if face_detected:
            dist_to_align = np.sqrt((target_denis_x_nose - denis_current_x)**2 + (target_denis_y_nose - denis_current_y)**2)
            if dist_to_align < ALIGNMENT_THRESHOLD:
                denis_controls_active = True; print("Kontrol diaktifkan!")
            else:
                cv2.putText(current_display_frame, "Arahkan hidung ke karakter untuk mulai",
                            (frame_width//2 - 250, frame_height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (200,200,255), 2, cv2.LINE_AA)

    # Logika pergerakan Denis (PUSAT HITBOX)
    if denis_spawned_safely and denis_controls_active:
        delta_x = effective_target_x - denis_current_x
        delta_y = effective_target_y - denis_current_y
        distance = np.sqrt(delta_x**2 + delta_y**2)
        next_center_x_candidate, next_center_y_candidate = denis_current_x, denis_current_y

        if distance > 1e-5 : # Hanya bergerak jika ada jarak (mencegah jitter saat target = posisi)
            if distance > DENIS_SPEED:
                move_x_norm = delta_x / distance; move_y_norm = delta_y / distance
                raw_step_x = move_x_norm * DENIS_SPEED; raw_step_y = move_y_norm * DENIS_SPEED
                step_x = int(np.sign(raw_step_x)) if abs(raw_step_x)<1.0 and abs(raw_step_x)>1e-5 else int(raw_step_x)
                step_y = int(np.sign(raw_step_y)) if abs(raw_step_y)<1.0 and abs(raw_step_y)>1e-5 else int(raw_step_y)
                next_center_x_candidate = denis_current_x + step_x
                next_center_y_candidate = denis_current_y + step_y
            elif distance > 1 : # Snap jika dekat (tapi tidak terlalu dekat untuk cegah getar)
                next_center_x_candidate = effective_target_x
                next_center_y_candidate = effective_target_y
        
        # Cek tabrakan X dan Y dengan hitbox lingkaran
        coll_x = check_collision_circle(next_center_x_candidate, denis_current_y, HITBOX_RADIUS, labirin_pil_resized_for_collision)
        if not coll_x: denis_current_x = next_center_x_candidate
        coll_y = check_collision_circle(denis_current_x, next_center_y_candidate, HITBOX_RADIUS, labirin_pil_resized_for_collision)
        if not coll_y: denis_current_y = next_center_y_candidate

    # Gambar VISUAL Denis
    denis_to_overlay_pil = denis_resized_pil_visual
    if denis_resized_pil_visual:
        if denis_spawned_safely and not denis_controls_active: # Redupkan
            try:
                enhancer = ImageEnhance.Brightness(denis_resized_pil_visual.copy())
                denis_to_overlay_pil = enhancer.enhance(0.5)
            except: denis_to_overlay_pil = denis_resized_pil_visual # Fallback
        
        draw_visual_x = denis_current_x - VISUAL_DENIS_WIDTH // 2
        draw_visual_y = denis_current_y - visual_denis_h // 2 # Gunakan visual_denis_h
        current_display_frame = overlay_image_alpha(current_display_frame, denis_to_overlay_pil, draw_visual_x, draw_visual_y)

        # DEBUG: Gambar hitbox (aktifkan jika perlu)
        # cv2.circle(current_display_frame, (int(denis_current_x), int(denis_current_y)), int(HITBOX_RADIUS), (0,255,0), 1)

    cv2.imshow('Dit Tolongin Dit Maze Filter', current_display_frame)
    if cv2.waitKey(5) & 0xFF == ord('q'): break

# --------------- CLEANUP ---------------
face_mesh.close(); cap.release(); cv2.destroyAllWindows()
if speech_thread.is_alive(): speech_thread.join(timeout=1.0)
print("Filter ditutup.")