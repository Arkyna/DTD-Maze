import cv2
import mediapipe as mp
import numpy as np

# Inisialisasi MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils # Untuk menggambar landmark (opsional)
mp_drawing_styles = mp.solutions.drawing_styles # Untuk gaya penggambaran (opsional)

# Fungsi untuk menempelkan gambar overlay (gambar test)
def overlay_image_alpha(img, img_overlay, x, y, alpha_mask):
    """Menempelkan img_overlay pada img di posisi x, y dengan alpha channel.
    alpha_mask adalah channel alpha dari img_overlay yang sudah di-resize.
    """
    # Mendapatkan dimensi gambar overlay
    h_overlay, w_overlay = img_overlay.shape[:2]

    # Mendapatkan region of interest (ROI) pada gambar utama
    # Pastikan ROI tidak keluar dari batas gambar utama
    y1, y2 = max(0, y), min(img.shape[0], y + h_overlay)
    x1, x2 = max(0, x), min(img.shape[1], x + w_overlay)

    # Sesuaikan gambar overlay dan alpha mask jika ROI lebih kecil
    roi_h = y2 - y1
    roi_w = x2 - x1
    img_overlay_resized = img_overlay[:roi_h, :roi_w]
    alpha_mask_resized = alpha_mask[:roi_h, :roi_w]

    if roi_h > 0 and roi_w > 0: # Pastikan ada area untuk di-overlay
        # Inverse alpha mask
        inv_alpha_mask = cv2.bitwise_not(alpha_mask_resized)

        # Hitamkan area overlay pada ROI di gambar utama
        img_bg_masked = cv2.bitwise_and(img[y1:y2, x1:x2], img[y1:y2, x1:x2], mask=inv_alpha_mask)

        # Ambil hanya region dari gambar overlay
        img_overlay_masked = cv2.bitwise_and(img_overlay_resized, img_overlay_resized, mask=alpha_mask_resized)

        # Tambahkan gambar overlay yang sudah di-mask ke ROI yang sudah di-mask
        # Pastikan kedua gambar memiliki jumlah channel yang sama sebelum add
        if img_bg_masked.shape[2] == 3 and img_overlay_masked.shape[2] == 4: # Jika overlay punya alpha, ambil BGR nya saja
            img_overlay_masked_bgr = img_overlay_masked[:,:,:3]
        elif img_bg_masked.shape[2] == 4 and img_overlay_masked.shape[2] == 3: # Jika background punya alpha, tambahkan alpha ke overlay
             img_overlay_masked_bgr = cv2.cvtColor(img_overlay_masked, cv2.COLOR_BGR2BGRA)
        else:
            img_overlay_masked_bgr = img_overlay_masked[:,:,:3] if img_overlay_masked.shape[2] == 4 else img_overlay_masked

        # Pastikan img_bg_masked dan img_overlay_masked_bgr memiliki tipe data dan channel yang sama
        # Biasanya error terjadi jika salah satu uint8 dan lainnya float, atau channel tidak cocok
        if img_bg_masked.shape[:2] == img_overlay_masked_bgr.shape[:2]: # Cek dimensi spasial
             # Safety check untuk jumlah channel
            if img_bg_masked.shape[2] != img_overlay_masked_bgr.shape[2]:
                # Jika gambar utama tidak punya alpha, dan overlay punya, kita convert overlay ke BGR
                if img_bg_masked.shape[2] == 3 and img_overlay_masked_bgr.shape[2] == 4:
                    img_overlay_masked_bgr = cv2.cvtColor(img_overlay_masked_bgr, cv2.COLOR_BGRA2BGR)
                # Tambahkan kondisi lain jika diperlukan

            if img_bg_masked.shape == img_overlay_masked_bgr.shape:
                 dst = cv2.add(img_bg_masked, img_overlay_masked_bgr)
                 img[y1:y2, x1:x2] = dst
            else:
                # Jika ada perbedaan channel setelah upaya konversi, print error
                print(f"Shape mismatch for add: BG {img_bg_masked.shape}, Overlay {img_overlay_masked_bgr.shape}")
        else:
            print(f"Spatial dimension mismatch for add: BG {img_bg_masked.shape[:2]}, Overlay {img_overlay_masked_bgr.shape[:2]}")

# --- PERSIAPAN GAMBAR TEST ---
# Contoh: test_image_original = cv2.imread('assets/denis_character.png', cv2.IMREAD_UNCHANGED)
try:
    test_image_original = cv2.imread('assets/denis.png.png', cv2.IMREAD_UNCHANGED)
    if test_image_original is None:
        raise FileNotFoundError("Gambar 'denis_character.png' tidak ditemukan di folder 'assets'.")
    # (Opsional) Resize jika perlu
    # desired_height = 50
    # aspect_ratio = test_image_original.shape[1] / test_image_original.shape[0]
    # desired_width = int(desired_height * aspect_ratio)
    # test_image_original = cv2.resize(test_image_original, (desired_width, desired_height))

except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Membuat placeholder darurat...")
    test_image_original = np.zeros((50, 50, 4), dtype=np.uint8)
    cv2.circle(test_image_original, (25,25), 20, (0,0,255,255), -1) # Lingkaran merah
except Exception as e:
    print(f"Error memuat gambar: {e}")
    # Fallback ke placeholder jika ada error lain
    test_image_original = np.zeros((50, 50, 4), dtype=np.uint8)
    cv2.circle(test_image_original, (25,25), 20, (0,0,255,255), -1) # Lingkaran merah
if test_image_original is None:
    print("Error: Tidak bisa membuat atau memuat gambar test.")
    # Jika gagal, buat placeholder sederhana lagi
    test_image_original = np.zeros((50, 50, 4), dtype=np.uint8)
    test_image_original[:, :, 0] = 0  # Blue
    test_image_original[:, :, 1] = 255 # Green
    test_image_original[:, :, 2] = 0  # Red
    test_image_original[:, :, 3] = 255 # Alpha (opaque)

# Ekstrak channel alpha jika ada, jika tidak, buat mask opak
if test_image_original.shape[2] == 4:
    test_image_bgr = test_image_original[:,:,:3]
    alpha_channel = test_image_original[:,:,3]
else:
    test_image_bgr = test_image_original
    alpha_channel = np.ones(test_image_bgr.shape[:2], dtype=test_image_bgr.dtype) * 255


# Landmark hidung (indeks 1 adalah salah satu titik di batang hidung bagian atas)
# Indeks lain yang bisa dicoba untuk hidung: 4 (ujung hidung), 5, 6, 195, 197, dll.
# Lihat visualisasi landmark MediaPipe Face Mesh untuk indeks yang lebih tepat:
# https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
NOSE_TIP_LANDMARK_INDEX = 1 # Coba ganti dengan 4 untuk ujung hidung


# Inisialisasi Video Capture dari webcam
cap = cv2.VideoCapture(0) # 0 untuk webcam utama

with mp_face_mesh.FaceMesh(
    max_num_faces=1,            # Hanya deteksi satu wajah
    refine_landmarks=True,      # Dapatkan landmark yang lebih detail (untuk mata, bibir, dll. Nanti berguna)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as face_mesh:

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Balik frame secara horizontal (efek cermin seperti di video call)
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        # Konversi warna BGR ke RGB (MediaPipe perlu RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Untuk meningkatkan performa, tandai frame sebagai tidak bisa ditulis (pass by reference)
        rgb_frame.flags.writeable = False
        results = face_mesh.process(rgb_frame)
        rgb_frame.flags.writeable = True # Kembalikan agar bisa diubah

        # Konversi kembali ke BGR untuk ditampilkan dengan OpenCV
        # frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR) # Akan dilakukan setelah overlay

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # --- 1. Menggambar semua landmark (OPSIONAL, untuk debugging) ---
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
                mp_drawing.draw_landmarks(
                   image=frame,
                   landmark_list=face_landmarks,
                   connections=mp_face_mesh.FACEMESH_IRISES, # Membutuhkan refine_landmarks=True
                   landmark_drawing_spec=None,
                   connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())


                # --- 2. Mendapatkan koordinat landmark hidung ---
                nose_landmark = face_landmarks.landmark[NOSE_TIP_LANDMARK_INDEX]

                # Koordinat landmark dinormalisasi (0.0 - 1.0)
                # Ubah ke koordinat piksel
                nose_x = int(nose_landmark.x * frame_width)
                nose_y = int(nose_landmark.y * frame_height)

                # Gambar lingkaran kecil di hidung (untuk debugging posisi)
                # cv2.circle(frame, (nose_x, nose_y), 5, (0, 255, 0), -1) # Hijau

                # --- 3. Menempatkan gambar test di hidung ---
                # Hitung posisi kiri atas untuk gambar test agar tengahnya ada di hidung
                img_h, img_w = test_image_bgr.shape[:2]
                pos_x = nose_x - img_w // 2
                pos_y = nose_y - img_h // 2

                overlay_image_alpha(frame, test_image_bgr, pos_x, pos_y, alpha_channel)


        # Tampilkan frame
        cv2.imshow('Deteksi Hidung MediaPipe - Test', frame)

        # Keluar jika tombol ESC ditekan
        if cv2.waitKey(5) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()