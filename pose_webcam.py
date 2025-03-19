from ultralytics import YOLO
import cv2
import numpy as np

# Load model YOLOv8 Pose
model = YOLO("intelligent-control-week5/yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(0)

# Daftar nama keypoints berdasarkan YOLOv8 Pose
keypoint_names = [
    "Nose", "Left Eye", "Right Eye", "Left Ear", "Right Ear",
    "Left Shoulder", "Right Shoulder", "Left Elbow", "Right Elbow",
    "Left Wrist", "Right Wrist", "Left Hip", "Right Hip",
    "Left Knee", "Right Knee", "Left Ankle", "Right Ankle"
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi pose
    results = model(frame)

    for result in results:
        annotated_frame = result.plot()  # Tambahkan anotasi pada frame
        
        # Ambil keypoints dan confidence scores
        keypoints = result.keypoints.xy.cpu().numpy()  # Koordinat keypoints
        confidences = result.keypoints.conf.cpu().numpy()  # Confidence score keypoints

        if len(keypoints) > 0:  # Pastikan ada keypoints yang terdeteksi
            for i, (x, y) in enumerate(keypoints[0]):  # Loop untuk setiap keypoint
                confidence_score = round(confidences[0][i] * 100, 1)  # Skor dalam persen
                
                if x > 0 and y > 0:  # Pastikan titik valid
                    x, y = int(x), int(y)  # Konversi ke integer
                    cv2.circle(annotated_frame, (x, y), 5, (0, 255, 0), -1)  # Gambar titik
                    cv2.putText(annotated_frame, f"{keypoint_names[i]}: {confidence_score}%", 
                                (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                                (255, 255, 255), 1, cv2.LINE_AA)  # Tambahkan teks

    # Tampilkan hasil dengan anotasi
    cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan sumber daya
cap.release()
cv2.destroyAllWindows()
