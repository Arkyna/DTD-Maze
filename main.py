import cv2
import mediapipe as mp
import numpy as np
import speech_recognition as sr
import pygame

# Inisialisasi pygame hehe
pygame.init()
screen = pygame.display.set_mode((640, 480))
pygame.display.set_caption("Dit Tolongin Dit Maze")
font = pygame.font.SysFont(None, 36)
clock = pygame.time.Clock()

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh

# Fungsi overlay gambar

def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    h_overlay, w_overlay = img_overlay.shape[:2]
    y1, y2 = max(0, y), min(img.shape[0], y + h_overlay)
    x1, x2 = max(0, x), min(img.shape[1], x + w_overlay)
    roi_h, roi_w = y2 - y1, x2 - x1
    img_overlay_resized = img_overlay[:roi_h, :roi_w]
    alpha_mask_resized = alpha_mask[:roi_h, :roi_w]
    if roi_h > 0 and roi_w > 0:
        inv_alpha_mask = cv2.bitwise_not(alpha_mask_resized)
        img_bg_masked = cv2.bitwise_and(img[y1:y2, x1:x2], img[y1:y2, x1:x2], mask=inv_alpha_mask)
        img_overlay_masked = cv2.bitwise_and(img_overlay_resized, img_overlay_resized, mask=alpha_mask_resized)
        img_overlay_masked_bgr = img_overlay_masked[:, :, :3]
        if img_bg_masked.shape[2] != img_overlay_masked_bgr.shape[2]:
            img_overlay_masked_bgr = cv2.cvtColor(img_overlay_masked, cv2.COLOR_BGRA2BGR)
        dst = cv2.add(img_bg_masked, img_overlay_masked_bgr)
        img[y1:y2, x1:x2] = dst

# Fungsi deteksi suara

def detect_keyword(keyword="dit tolongin dit"):
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("[Speech] Listening for keyword...")
        try:
            audio = recognizer.listen(source, timeout=3)
            text = recognizer.recognize_google(audio, language="id-ID")
            print("[Speech] Detected:", text)
            return keyword.lower() in text.lower()
        except:
            return False

# Load gambar Denis
try:
    test_image_original = cv2.imread('assets/denis.png', cv2.IMREAD_UNCHANGED)
    if test_image_original is None:
        raise FileNotFoundError("denis.png not found")
except:
    test_image_original = np.zeros((50, 50, 4), dtype=np.uint8)
    cv2.circle(test_image_original, (25, 25), 20, (0, 0, 255, 255), -1)

if test_image_original.shape[2] == 4:
    test_image_bgr = test_image_original[:, :, :3]
    alpha_channel = test_image_original[:, :, 3]
else:
    test_image_bgr = test_image_original
    alpha_channel = np.ones(test_image_bgr.shape[:2], dtype=test_image_bgr.dtype) * 255

NOSE_TIP_LANDMARK_INDEX = 1
cap = cv2.VideoCapture(0)

petunjuk_aktif = False
listen_trigger = False

with mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_s:
                    listen_trigger = True

        success, frame = cap.read()
        if not success:
            continue

        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                nose = face_landmarks.landmark[NOSE_TIP_LANDMARK_INDEX]
                nose_x = int(nose.x * frame_width)
                nose_y = int(nose.y * frame_height)
                pos_x = nose_x - test_image_bgr.shape[1] // 2
                pos_y = nose_y - test_image_bgr.shape[0] // 2
                overlay_image_alpha(frame, test_image_bgr, pos_x, pos_y, alpha_channel)

                if listen_trigger:
                    if detect_keyword():
                        petunjuk_aktif = True
                    listen_trigger = False

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_surface = pygame.surfarray.make_surface(np.rot90(frame_rgb))
        screen.blit(pygame.transform.scale(frame_surface, (640, 480)), (0, 0))

        if petunjuk_aktif:
            text = font.render("PETUNJUK AKTIF!", True, (255, 0, 0))
            screen.blit(text, (50, 50))

        pygame.display.flip()
        clock.tick(30)

cap.release()
pygame.quit()
