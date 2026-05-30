import cv2
from ultralytics import YOLO

# load model hasil training
model = YOLO('Hasil_Training_CCTV/Model_V13/weights/best.pt')

video_path = 'data/siangmalamtest.3gp'
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Tidak bisa membuka video.")
    exit()

fps = cap.get(cv2.CAP_PROP_FPS)
# batasi 5 menit aja buat testing
maksimal_frame = int(fps * 5 * 60)

print(f"FPS: {fps}, proses {maksimal_frame} frame (5 menit pertama)")

frame_count = 0

while cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame_count >= maksimal_frame:
        break

    results = model(frame, conf=0.3, verbose=False)
    annotated_frame = results[0].plot()

    display_frame = cv2.resize(annotated_frame, (960, 720))
    cv2.imshow("Tes YOLO CCTV", display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Dihentikan manual.")
        break

    frame_count += 1

cap.release()
cv2.destroyAllWindows()
print("Selesai.")
