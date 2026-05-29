from roboflow import Roboflow
from ultralytics import YOLO


#kode dari roboflow
rf = Roboflow(api_key="coEQt6UNclWCOcbIpqKm")
project = rf.workspace("tams-workspace-l1vdi").project("cctv-vehicle-detection-comvis")
version = project.version(1)
dataset = version.download("yolov8")
                
#load model dasar YOLOv8 (menggunakan versi 'nano' (n) agar ringan dan FPS tinggi saat digabung ByteTrack)
model = YOLO('yolo11n.pt') 

#training dan tuning
if __name__ == '__main__':
    print("Memulai training YOLOv8 pada GPU RTX 3050...")
    
    #dataset.location otomatis mengambil path folder dataset yang baru didownload
    model.train(
        data=f"{dataset.location}/data.yaml", 
        epochs=100,                           #Maksimal 100 putaran belajar
        imgsz=640,                            #Ukuran gambar sesuai resize di Roboflow
        batch=8,                              #Agar aman untuk VRAM 6GB
        device=0,                             #Kode '0' berarti memaksa pakai GPU NVIDIA
        patience=50,                          #Fitur berhenti otomatis jika akurasi sudah mentok
        workers=2,                            #Membantu CPU load data (set 2 aman untuk Windows)
        project="Hasil_Training_CCTV",        #Nama folder utama hasil training
        name="Model_V1",                      #Nama sub-folder
        save=True                             #Memastikan model terbaik disimpan
    )
    
    print("Training Selesai (folder output: Hasil_Training_CCTV/Model_V1/weights/)")