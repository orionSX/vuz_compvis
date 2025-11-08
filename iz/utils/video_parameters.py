import cv2
import numpy as np
import os
from video_file_manager import VideoFileManager


def get_video_properties(video_path):
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("Ошибка открытия видео")
        return None
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration = frame_count / fps if fps > 0 else 0
    
    fourcc_code = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = fourcc_code.to_bytes(4, 'little').decode('ascii')

    cap.release()
    
    return {
        'fps': fps,
        'frame_count': frame_count,
        'resolution': (width, height),
        'duration_sec': duration,
        'codec_fourcc': codec
    }


def calculate_motion_intensity(video_path, sample_frames):
    cap = cv2.VideoCapture(video_path)
    prev_frame = None
    motion_values = []
    
    frame_interval = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT) / sample_frames))
    
    for i in range(sample_frames):
        ret, frame = cap.read()
        if not ret:
            break
            
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * frame_interval)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if prev_frame is not None:
            diff = cv2.absdiff(prev_frame, gray)
            motion = np.mean(diff)
            motion_values.append(motion)
        
        prev_frame = gray
    
    cap.release()
    
    if not motion_values:
        return 0
    
    avg_motion = np.mean(motion_values)
    

    if avg_motion < 5:
        intensity_level = "очень низкий"
    elif avg_motion < 15:
        intensity_level = "низкий"
    elif avg_motion < 30:
        intensity_level = "средний"
    elif avg_motion < 50:
        intensity_level = "высокий"
    else:
        intensity_level = "очень высокий"
    
    return {
        'raw_intensity': avg_motion,
        'intensity_level': intensity_level,
        'max_motion': np.max(motion_values),
        'std_motion': np.std(motion_values)
    }


def main():
    video_files = VideoFileManager.get_video_files(r"videos")
    
    print(f"Найдено видеофайлов: {len(video_files)}\n")
    
    for i, video_path in enumerate(video_files, 1):
        print(f"=== Видео {i}/{len(video_files)}: {os.path.basename(video_path)} ===")
        
        props_video = get_video_properties(video_path)
        if props_video is None:
            print("Ошибка: не удалось открыть видеофайл\n")
            continue
        
        print(f"FPS: {props_video['fps']:.2f}")
        print(f"Общее количество кадров: {props_video['frame_count']}")
        intensity_video = calculate_motion_intensity(video_path, props_video['frame_count'] // 10)
        
        print(f"Средняя интенсивность: {intensity_video['raw_intensity']:.2f}, уровень интенсивности: {intensity_video['intensity_level']}")
        print(f"Разрешение: {props_video['resolution']}")
        print(f"Длительность: {props_video['duration_sec']:.2f} сек")
        print(f"Кодек: {props_video['codec_fourcc']}")
        print()


if __name__ == "__main__":
    main()
