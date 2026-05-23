import subprocess
import os

def extract_frames_gpu(video_paths, output_folder, interval_seconds):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_path in video_paths:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_pattern = os.path.join(output_folder, f"{video_name}_frame_%04d.jpg")
        
        print(f"\nMemproses '{video_name}' dengan GPU...")
        
        # Command FFmpeg memanfaatkan CUDA/NVDEC
        # -hwaccel cuda: Gunakan hardware acceleration NVIDIA
        # -vf fps=1/15: Ambil 1 frame setiap 15 detik
        # -q:v 2: Kualitas JPEG tinggi (range 1-31, makin kecil makin bagus)
        command = [
            'ffmpeg', 
            '-y', # Overwrite file jika ada
            '-hwaccel', 'cuda', 
            '-i', video_path, 
            '-vf', f'fps=1/{interval_seconds}', 
            '-q:v', '2', 
            output_pattern
        ]
        
        # Jalankan command di background
        subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        print(f"Selesai memproses '{video_name}'.")

video_files = ["data/pagisiang.3gp", "data/siangmalam.3gp"]
extract_frames_gpu(video_files, "dataset_raw_roboflow", interval_seconds=20)