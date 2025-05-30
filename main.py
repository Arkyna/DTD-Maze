import pygame
import sys
import cv2
import mediapipe as mp
import speech_recognition as sr
import threading
import os

pygame.init()

WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

BASE_ASSET_PATH = ".//assets"
MAZE_IMAGE_PATH_NO_HINT = os.path.join(BASE_ASSET_PATH, "labirin_noBG.png")
MAZE_IMAGE_PATH_HINT = os.path.join(BASE_ASSET_PATH, "hint_labirin.png")
CURRENT_DISPLAY_MAZE_PATH = MAZE_IMAGE_PATH_NO_HINT

PLAYER_OVERLAY_PATH = os.path.join(BASE_ASSET_PATH, "denis_diam.png")
player_overlay_image = None
PLAYER_OVERLAY_TARGET_BOX_SIZE = (110, 110)
actual_overlay_width, actual_overlay_height = 0, 0

MAZE_SCALE_FACTOR = 1.5
WHITE, BLACK, RED, BLUE, GREY = (255,255,255), (0,0,0), (255,0,0), (0,0,255), (200,200,200)
GREEN_NEUTRAL, UI_BACKGROUND, YELLOW_HINT_MSG = (0,255,0), WHITE, (200,200,0)

screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Game Labirin - Kontrol Hidung & Suara!")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)
if not cap.isOpened(): print("Tidak dapat membuka kamera.")

initial_nose_x, initial_nose_y = 0, 0
initial_nose_position_set = False
NEUTRAL_ZONE_RADIUS = 35

recognizer = sr.Recognizer()
microphone = None
try:
    microphone = sr.Microphone()
except Exception as e:
    print(f"Error inisialisasi microphone: {e}. Speech recognition tidak akan berfungsi.")

hint_requested_by_voice, hint_active = False, False
listening_status_message = ""

maze_display_image = None
maze_collision_mask = None
actual_maze_width, actual_maze_height = 0, 0
MAZE_OFFSET_X, MAZE_OFFSET_Y = 0, 0
maze_boundary_rect = None

# --- Variabel Timer ---
start_time = 0  # Akan diinisialisasi setelah aset dimuat
elapsed_time_str = "Waktu: 00:00"
final_elapsed_time_str = "" # Untuk menyimpan waktu saat menang
game_started_timer = False # Flag untuk menandakan timer sudah dimulai

def load_maze_assets(image_path_to_load, create_collision_mask_from_this_image):
    global maze_display_image, maze_collision_mask, actual_maze_width, actual_maze_height
    global MAZE_OFFSET_X, MAZE_OFFSET_Y, maze_boundary_rect, CURRENT_DISPLAY_MAZE_PATH

    CURRENT_DISPLAY_MAZE_PATH = image_path_to_load
    try:
        loaded_surface = pygame.image.load(image_path_to_load).convert_alpha()
        original_width = loaded_surface.get_width()
        original_height = loaded_surface.get_height()

        scaled_width = int(original_width * MAZE_SCALE_FACTOR)
        scaled_height = int(original_height * MAZE_SCALE_FACTOR)

        maze_display_image = pygame.transform.scale(loaded_surface, (scaled_width, scaled_height))
        actual_maze_width = maze_display_image.get_width()
        actual_maze_height = maze_display_image.get_height()

        MAZE_OFFSET_X = (WINDOW_WIDTH - actual_maze_width) // 2
        MAZE_OFFSET_Y = (WINDOW_HEIGHT - actual_maze_height) // 2

        if actual_maze_width > 0 and actual_maze_height > 0:
            maze_boundary_rect = pygame.Rect(MAZE_OFFSET_X, MAZE_OFFSET_Y, actual_maze_width, actual_maze_height)
        else:
            maze_boundary_rect = pygame.Rect(0,0,0,0)

        if create_collision_mask_from_this_image:
            maze_collision_mask = pygame.mask.from_surface(maze_display_image, threshold=120)
            print(f"Masker tabrakan diperbarui dari '{image_path_to_load}'.")

        print(f"Gambar '{image_path_to_load}' (untuk tampilan) berhasil dimuat.")
        return True

    except pygame.error as e:
        print(f"Tidak dapat memuat gambar labirin '{image_path_to_load}': {e}")
        maze_display_image = pygame.Surface((100,100))
        maze_display_image.fill(BLACK)
        actual_maze_width = 100; actual_maze_height = 100
        MAZE_OFFSET_X = (WINDOW_WIDTH - actual_maze_width) // 2
        MAZE_OFFSET_Y = (WINDOW_HEIGHT - actual_maze_height) // 2
        maze_boundary_rect = pygame.Rect(MAZE_OFFSET_X, MAZE_OFFSET_Y, actual_maze_width, actual_maze_height)
        if create_collision_mask_from_this_image:
            maze_collision_mask = None
        return False

player_size, player_speed = 15, 2
player_start_x_ratio, player_start_y_ratio = 0.46, 0.00
player_surface = pygame.Surface((player_size, player_size), pygame.SRCALPHA)
player_surface.fill(RED)
player_mask = pygame.mask.from_surface(player_surface)

try:
    player_overlay_image_original = pygame.image.load(PLAYER_OVERLAY_PATH).convert_alpha()
    original_img_width = player_overlay_image_original.get_width()
    original_img_height = player_overlay_image_original.get_height()
    target_box_w, target_box_h = PLAYER_OVERLAY_TARGET_BOX_SIZE

    if original_img_width == 0 or original_img_height == 0:
        raise ValueError("Ukuran gambar asli adalah nol.")

    scale_w = target_box_w / original_img_width
    scale_h = target_box_h / original_img_height
    scale_factor = min(scale_w, scale_h)

    new_overlay_width = int(original_img_width * scale_factor)
    new_overlay_height = int(original_img_height * scale_factor)

    if new_overlay_width > 0 and new_overlay_height > 0:
        player_overlay_image = pygame.transform.smoothscale(player_overlay_image_original, (new_overlay_width, new_overlay_height))
        actual_overlay_width, actual_overlay_height = player_overlay_image.get_size()
        print(f"Gambar overlay pemain '{PLAYER_OVERLAY_PATH}' berhasil dimuat dan di-resize ke ({actual_overlay_width}, {actual_overlay_height}) dengan menjaga rasio aspek.")
    else:
        print(f"Gagal menghitung ukuran baru untuk overlay. Menggunakan ukuran asli atau tidak ada overlay.")
        if original_img_width <= target_box_w and original_img_height <= target_box_h and original_img_width > 0 and original_img_height > 0:
            player_overlay_image = player_overlay_image_original
            actual_overlay_width, actual_overlay_height = player_overlay_image.get_size()
        else:
            player_overlay_image = None
            actual_overlay_width, actual_overlay_height = 0,0
except Exception as e:
    print(f"Gagal memuat atau memproses gambar overlay pemain '{PLAYER_OVERLAY_PATH}': {e}")
    player_overlay_image = None
    actual_overlay_width, actual_overlay_height = 0, 0

if not load_maze_assets(MAZE_IMAGE_PATH_NO_HINT, True):
    print("KRITIKAL: Gagal memuat labirin awal dan maskernya. Game mungkin tidak berfungsi dengan benar.")
else:
    # Mulai timer setelah aset utama berhasil dimuat
    start_time = pygame.time.get_ticks()
    game_started_timer = True


player_rect = pygame.Rect(
    MAZE_OFFSET_X + int(actual_maze_width * player_start_x_ratio),
    MAZE_OFFSET_Y + int(actual_maze_height * player_start_y_ratio),
    player_size, player_size
)

initial_player_center_x, initial_player_center_y = player_rect.centerx, player_rect.centery
player_has_moved_from_spawn = False
MIN_DISTANCE_FROM_SPAWN_FOR_WIN = player_size * 2

font = pygame.font.Font(None, 74)
title_font = pygame.font.Font(None, 50) # Font untuk judul "KAMU MENANG!"
info_font = pygame.font.Font(None, 38) # Font untuk info waktu
small_font = pygame.font.Font(None, 36)
micro_font = pygame.font.Font(None, 24)

def check_maze_collision_with_masks(p_rect, p_mask, m_collision_mask, m_offset_x, m_offset_y):
    if not m_collision_mask or not p_mask:
        return False
    offset_x = p_rect.x - m_offset_x
    offset_y = p_rect.y - m_offset_y
    collision_point = m_collision_mask.overlap(p_mask, (offset_x, offset_y))
    return True if collision_point else False

def listen_for_hint_command():
    global hint_requested_by_voice, listening_status_message, running
    if not microphone:
        listening_status_message = "Mikrofon tidak tersedia."
        return

    print("Thread speech recognition dimulai...")
    try:
        with microphone as source:
            print("Kalibrasi suara sekitar (1 detik)...")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            print("Kalibrasi selesai. Katakan 'dit tolongin dit'.")
            listening_status_message = "Katakan 'dit tolongin dit' untuk hint."
    except Exception as e:
        print(f"Error kalibrasi mic: {e}")
        listening_status_message = "Error kalibrasi mic."

    while running:
        if not microphone:
            listening_status_message = "Mikrofon error. Coba lagi nanti."
            pygame.time.wait(1000)
            continue
        if hint_active: # Jika hint sudah aktif, tidak perlu mendengarkan lagi
            # Cek apakah pesan hint sudah aktif, jika belum, set.
            # Ini untuk menghindari pesan "Mendengarkan..." muncul lagi setelah hint aktif.
            if "Hint sudah aktif" not in listening_status_message:
                 listening_status_message = "Hint sudah aktif."
            pygame.time.wait(1000)
            continue

        current_listening_msg = "Mendengarkan..."
        try:
            if not ("Mendengarkan..." in listening_status_message or "Hint diminta" in listening_status_message or "Hint sudah aktif" in listening_status_message):
                listening_status_message = current_listening_msg

            with microphone as source:
                audio = recognizer.listen(source, timeout=3, phrase_time_limit=4)

            listening_status_message = "Memproses suara..."
            command = recognizer.recognize_google(audio, language="id-ID").lower()
            print(f"Anda mengatakan: {command}")
            listening_status_message = f"Terdeteksi: {command[:30]}..."

            if "dit tolongin dit" in command and not hint_active:
                print("Perintah hint terdeteksi!")
                hint_requested_by_voice = True
                listening_status_message = "Hint diminta! Mengaktifkan..."
            pygame.time.wait(200)

        except sr.WaitTimeoutError:
            if not "Katakan" in listening_status_message and not "Hint sudah aktif" in listening_status_message : listening_status_message = "Katakan 'dit tolongin dit'..."
        except sr.UnknownValueError:
            if not "Tidak dapat mengenali" in listening_status_message and not "Hint sudah aktif" in listening_status_message : listening_status_message = "Tidak dapat mengenali suara."
        except sr.RequestError as e:
            print(f"API Error: {e}"); listening_status_message = "Error koneksi speech API."
        except Exception as e:
            print(f"Error speech: {e}"); listening_status_message = "Error speech tidak diketahui."

        if not running: break
    print("Thread speech recognition dihentikan.")

running = True
game_won = False
clock = pygame.time.Clock()

if not maze_display_image or not maze_collision_mask:
    print("PERHATIAN: Labirin atau masker tabrakan tidak dimuat dengan benar. Ini bisa menyebabkan error.")

if microphone:
    listener_thread = threading.Thread(target=listen_for_hint_command, daemon=True)
    listener_thread.start()
else:
    listening_status_message = "Speech recognition tidak aktif (mikrofon error)."

cam_window_name = 'Kontrol Hidung - Kalibrasi & Kuadran'

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    if hint_requested_by_voice and not hint_active:
        print("Mengaktifkan hint...")
        if load_maze_assets(MAZE_IMAGE_PATH_HINT, False):
            hint_active = True
            print(f"Labirin hint dimuat untuk TAMPILAN. Posisi pemain TETAP. Masker tabrakan dari labirin asli.")
            listening_status_message = "Hint sudah aktif." # Langsung set status
        else:
            print("Gagal memuat labirin hint.")
            listening_status_message = "Gagal load hint."
        hint_requested_by_voice = False

    dx_nose, dy_nose = 0, 0
    if cap.isOpened():
        success, frame = cap.read()
        if success:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cam_h, cam_w, _ = frame.shape
            results = face_mesh.process(frame_rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    nose_tip = face_landmarks.landmark[1]
                    current_nose_x = int(nose_tip.x * cam_w)
                    current_nose_y = int(nose_tip.y * cam_h)
                    cv2.circle(frame, (current_nose_x, current_nose_y), 5, (0, 255, 255), -1)

                    if not initial_nose_position_set:
                        initial_nose_x, initial_nose_y = current_nose_x, current_nose_y
                        initial_nose_position_set = True
                        print(f"Kalibrasi hidung di: ({initial_nose_x}, {initial_nose_y})")

                    if initial_nose_position_set:
                        cv2.rectangle(frame, (initial_nose_x - NEUTRAL_ZONE_RADIUS, initial_nose_y - NEUTRAL_ZONE_RADIUS),
                                      (initial_nose_x + NEUTRAL_ZONE_RADIUS, initial_nose_y + NEUTRAL_ZONE_RADIUS), GREEN_NEUTRAL, 2)
                        cv2.line(frame, (initial_nose_x, 0), (initial_nose_x, cam_h), (255,0,0),1)
                        cv2.line(frame, (0, initial_nose_y), (cam_w, initial_nose_y), (255,0,0),1)

                        is_in_neutral = (initial_nose_x - NEUTRAL_ZONE_RADIUS < current_nose_x < initial_nose_x + NEUTRAL_ZONE_RADIUS and
                                         initial_nose_y - NEUTRAL_ZONE_RADIUS < current_nose_y < initial_nose_y + NEUTRAL_ZONE_RADIUS)

                        if is_in_neutral:
                            dx_nose, dy_nose = 0, 0
                        else:
                            if current_nose_x > initial_nose_x + NEUTRAL_ZONE_RADIUS and current_nose_y < initial_nose_y - NEUTRAL_ZONE_RADIUS:
                                dx_nose, dy_nose = player_speed, -player_speed
                            elif current_nose_x < initial_nose_x - NEUTRAL_ZONE_RADIUS and current_nose_y < initial_nose_y - NEUTRAL_ZONE_RADIUS:
                                dx_nose, dy_nose = -player_speed, -player_speed
                            elif current_nose_x < initial_nose_x - NEUTRAL_ZONE_RADIUS and current_nose_y > initial_nose_y + NEUTRAL_ZONE_RADIUS:
                                dx_nose, dy_nose = -player_speed, player_speed
                            elif current_nose_x > initial_nose_x + NEUTRAL_ZONE_RADIUS and current_nose_y > initial_nose_y + NEUTRAL_ZONE_RADIUS:
                                dx_nose, dy_nose = player_speed, player_speed
                            elif current_nose_x > initial_nose_x + NEUTRAL_ZONE_RADIUS: dx_nose = player_speed
                            elif current_nose_x < initial_nose_x - NEUTRAL_ZONE_RADIUS: dx_nose = -player_speed
                            elif current_nose_y < initial_nose_y - NEUTRAL_ZONE_RADIUS: dy_nose = -player_speed
                            elif current_nose_y > initial_nose_y + NEUTRAL_ZONE_RADIUS: dy_nose = player_speed
                    break
            else:
                if initial_nose_position_set: dx_nose, dy_nose = 0,0
                else: cv2.putText(frame, "Posisikan hidung di tengah", (cam_w//2-150, cam_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, WHITE, 2)
            cv2.imshow(cam_window_name, frame)

    # --- Logika Game (pergerakan, tabrakan, menang) & Timer ---
    if game_started_timer and not game_won: # Update timer jika game berjalan dan belum menang
        current_ticks = pygame.time.get_ticks()
        elapsed_milliseconds = current_ticks - start_time
        total_seconds = elapsed_milliseconds // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        elapsed_time_str = f"Waktu: {minutes:02}:{seconds:02}"

    if not game_won:
        original_player_x, original_player_y = player_rect.x, player_rect.y

        if dx_nose != 0:
            player_rect.x += dx_nose
            if maze_collision_mask and check_maze_collision_with_masks(player_rect, player_mask, maze_collision_mask, MAZE_OFFSET_X, MAZE_OFFSET_Y):
                player_rect.x = original_player_x

        if dy_nose != 0:
            player_rect.y += dy_nose
            if maze_collision_mask and check_maze_collision_with_masks(player_rect, player_mask, maze_collision_mask, MAZE_OFFSET_X, MAZE_OFFSET_Y):
                player_rect.y = original_player_y

        if maze_boundary_rect : player_rect.clamp_ip(maze_boundary_rect)

        if not player_has_moved_from_spawn:
            current_center_vec = pygame.math.Vector2(player_rect.centerx, player_rect.centery)
            initial_center_vec = pygame.math.Vector2(initial_player_center_x, initial_player_center_y)
            if current_center_vec.distance_to(initial_center_vec) > player_size * 0.5:
                player_has_moved_from_spawn = True
                print("Pemain bergerak dari spawn awal.")

        if player_has_moved_from_spawn and maze_boundary_rect:
            on_perimeter = (player_rect.left == maze_boundary_rect.left or
                            player_rect.right == maze_boundary_rect.right or
                            player_rect.top == maze_boundary_rect.top or
                            player_rect.bottom == maze_boundary_rect.bottom) and \
                           (player_rect.width > 0 and player_rect.height > 0)

            if on_perimeter:
                dist_from_spawn_vec = pygame.math.Vector2(player_rect.centerx, player_rect.centery).distance_to(pygame.math.Vector2(initial_player_center_x, initial_player_center_y))
                if dist_from_spawn_vec > MIN_DISTANCE_FROM_SPAWN_FOR_WIN:
                    if not game_won: # Hanya set sekali saat menang
                        game_won = True
                        print(f"MENANG! Jarak dari spawn: {dist_from_spawn_vec:.1f}")
                        # Catat waktu final saat menang
                        if game_started_timer:
                            final_elapsed_milliseconds = pygame.time.get_ticks() - start_time
                            final_total_seconds = final_elapsed_milliseconds // 1000
                            final_minutes = final_total_seconds // 60
                            final_seconds = final_total_seconds % 60
                            final_elapsed_time_str = f"{final_minutes:02}:{final_seconds:02}"
                            elapsed_time_str = f"Waktu: {final_elapsed_time_str}" # Freeze tampilan timer utama


    # --- Proses Menggambar (Rendering) ---
    screen.fill(WHITE)
    pygame.draw.rect(screen, UI_BACKGROUND, (0, 0, WINDOW_WIDTH, MAZE_OFFSET_Y))
    pygame.draw.rect(screen, UI_BACKGROUND, (0, MAZE_OFFSET_Y + actual_maze_height, WINDOW_WIDTH, WINDOW_HEIGHT - (MAZE_OFFSET_Y + actual_maze_height)))
    pygame.draw.rect(screen, UI_BACKGROUND, (0, MAZE_OFFSET_Y, MAZE_OFFSET_X, actual_maze_height))
    pygame.draw.rect(screen, UI_BACKGROUND, (MAZE_OFFSET_X + actual_maze_width, MAZE_OFFSET_Y, WINDOW_WIDTH - (MAZE_OFFSET_X + actual_maze_width), actual_maze_height))

    # Tampilkan Timer
    time_text_surface = small_font.render(elapsed_time_str, True, BLACK)
    screen.blit(time_text_surface, (10, 10))

    speech_color = YELLOW_HINT_MSG if "hint" in listening_status_message.lower() or "diminta" in listening_status_message.lower() or "aktif" in listening_status_message.lower() else BLACK
    speech_status_surf = micro_font.render(listening_status_message, True, speech_color)
    screen.blit(speech_status_surf, (10, WINDOW_HEIGHT - 30))

    hint_btn_rect = pygame.Rect(WINDOW_WIDTH - 110, 10, 100, 30)
    btn_color = (100,100,100) if hint_active else (0,200,0)
    pygame.draw.rect(screen, btn_color, hint_btn_rect)
    hint_txt_surf = small_font.render("Hint" if not hint_active else "Hint ON", True, BLACK)
    screen.blit(hint_txt_surf, hint_txt_surf.get_rect(center=hint_btn_rect.center))

    if maze_display_image:
        screen.blit(maze_display_image, (MAZE_OFFSET_X, MAZE_OFFSET_Y))
    else:
        if actual_maze_width > 0 and actual_maze_height > 0:
            placeholder_rect = pygame.Rect(MAZE_OFFSET_X, MAZE_OFFSET_Y, actual_maze_width, actual_maze_height)
            pygame.draw.rect(screen, BLACK, placeholder_rect, 2)
            text_surface = small_font.render("Error: Labirin Gagal Dimuat", True, BLACK)
            screen.blit(text_surface, text_surface.get_rect(center=placeholder_rect.center))

    if player_overlay_image and actual_overlay_width > 0 and actual_overlay_height > 0 :
        overlay_x = player_rect.centerx - actual_overlay_width // 2
        overlay_y = player_rect.centery - actual_overlay_height // 2
        screen.blit(player_overlay_image, (overlay_x, overlay_y))
    else:
        pygame.draw.rect(screen, RED, player_rect)

    if game_won:
        # Area latar belakang untuk pesan menang
        padding = 20
        popup_width = 400
        popup_height = 150 # Cukup untuk dua baris teks
        popup_rect = pygame.Rect((WINDOW_WIDTH - popup_width) // 2, (WINDOW_HEIGHT - popup_height) // 2, popup_width, popup_height)

        pygame.draw.rect(screen, GREY, popup_rect, border_radius=15) # Latar belakang abu-abu dengan sudut bulat
        pygame.draw.rect(screen, BLUE, popup_rect, 3, border_radius=15) # Border biru

        # Teks "KAMU MENANG!"
        win_title_surface = title_font.render("KAMU MENANG!", True, BLUE)
        win_title_rect = win_title_surface.get_rect(center=(popup_rect.centerx, popup_rect.centery - 25)) # Sedikit ke atas

        # Teks "Waktu: MM:SS"
        win_time_surface = info_font.render(f"Waktu Selesai: {final_elapsed_time_str}", True, BLACK)
        win_time_rect = win_time_surface.get_rect(center=(popup_rect.centerx, popup_rect.centery + 25)) # Sedikit ke bawah

        screen.blit(win_title_surface, win_title_rect)
        screen.blit(win_time_surface, win_time_rect)


    pygame.display.flip()

    key_cv = cv2.waitKey(1) & 0xFF
    if key_cv == 27:
        running = False

    clock.tick(30)

if cap.isOpened(): cap.release()
cv2.destroyAllWindows()
pygame.quit()
print("Game ditutup.")
sys.exit()
