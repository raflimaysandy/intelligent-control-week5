from ultralytics import YOLO
 # Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")
 # Deteksi pose pada gambar
results = model("C:\\Rafli\\Semester 6\\Prak.Kontrol Cerdas\\week5\\intelligent-control-week5\\foto.jpeg", show=True)
# Simpan hasil
results[0].save("pose_result.jpg")