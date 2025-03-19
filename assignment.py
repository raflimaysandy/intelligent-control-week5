from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

# Load model YOLOv8 Pose
model = YOLO("intelligent-control-week5/yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Kamera tidak dapat dibuka.")
    exit()

# Daftar nama keypoints tubuh berdasarkan YOLOv8 Pose
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera.")
        break

    # Konversi ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi pose tubuh dengan YOLOv8
    results = model(frame)

    for result in results:
        annotated_frame = result.plot()  # Tambahkan anotasi pada frame

        keypoints = result.keypoints.xy.numpy()  # Koordinat keypoints
        confidences = result.keypoints.conf.numpy()  # Confidence score keypoints

        for i, (person, conf) in enumerate(zip(keypoints, confidences)):
            if person.shape[0] >= 17:  # Pastikan jumlah keypoints cukup
                # Ganti titik (0,0) yang salah dengan NaN
                person = np.where(person > 0, person, np.nan)

                # Fungsi menggambar garis hanya jika kedua titik valid
                def draw_line(p1, p2, color=(0, 255, 0), thickness=2):
                    if p1 is not None and p2 is not None and not np.isnan(p1).any() and not np.isnan(p2).any():
                        cv2.line(annotated_frame, tuple(map(int, p1)), tuple(map(int, p2)), color, thickness)

                # Gambar titik-titik keypoints dengan label dan confidence
                for j, (x, y) in enumerate(person):
                    if not np.isnan(x) and not np.isnan(y):
                        x, y = int(x), int(y)
                        confidence_score = round(conf[j] * 100, 1)  # Skor dalam persen
                        
                        cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)  # Gambar titik
                        cv2.putText(annotated_frame, f"{keypoint_names[j]}: {confidence_score}%", 
                                    (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                    (255, 255, 255), 1, cv2.LINE_AA)  # Tambahkan teks

                # Koneksi antar keypoints untuk membentuk kerangka tubuh
                connections = [
                    (0, 5), (0, 6), (5, 6), (5, 11), (6, 12), (11, 12),  # Badan
                    (5, 7), (7, 9), (6, 8), (8, 10),  # Tangan
                    (11, 13), (13, 15), (12, 14), (14, 16)  # Kaki
                ]

                for p1, p2 in connections:
                    draw_line(person[p1], person[p2])

    # Deteksi tangan dengan MediaPipe
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Gambar titik pada jari-jari tangan
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

            # Gambar garis tangan menggunakan MediaPipe
            mp_draw.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Pose & Hand Estimation", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
