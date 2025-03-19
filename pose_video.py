from ultralytics import YOLO
 # Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")
 # Deteksi pose pada video
results = model("C:\\Rafli\\Semester 6\\Prak.Kontrol Cerdas\\week5\\intelligent-control-week5\\video.mp4", save=True, show=True)