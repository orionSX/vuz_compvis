import os


class VideoFileManager:
    VIDEO_EXTENSIONS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v']
    
    @staticmethod
    def get_video_files(folder_path):
        if not os.path.exists(folder_path):
            print(f"Папка {folder_path} не существует!")
            return []
        
        video_files = []
        for file in os.listdir(folder_path):
            if any(file.lower().endswith(ext) for ext in VideoFileManager.VIDEO_EXTENSIONS):
                video_files.append(os.path.join(folder_path, file))
        
        return sorted(video_files)
    
    @staticmethod
    def print_video_list(video_files):
        print(f"Найдено видеофайлов: {len(video_files)}\n")
        for i, video_path in enumerate(video_files, 1):
            filename = os.path.basename(video_path)
            print(f"{i:2d}. {filename}")
        print()
    
    @staticmethod
    def save_processed_video(tracker, video_path, output_dir, tracker_name):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_filename = f"{base_name}_{tracker_name}_tracked.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        tracker.track_from_video(video_path, output_path)
        
        return output_path
    
    @staticmethod
    def create_tracker_output_dir(base_dir, tracker_name):
        output_dir = os.path.join(base_dir, f"{tracker_name}_videos")
        os.makedirs(output_dir, exist_ok=True)
        return output_dir
    
    @staticmethod
    def get_video_info(video_path):
        if not os.path.exists(video_path):
            return None
        
        filename = os.path.basename(video_path)
        size = os.path.getsize(video_path) / (1024 * 1024)
        return {
            'filename': filename,
            'size_mb': round(size, 2),
            'full_path': video_path
        }
