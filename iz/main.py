import os
from utils import VideoFileManager
from cv2_trackers import *


def main():
    input_videos_dir = "videos"
    output_base_dir = "tracking_results"

    video_files = VideoFileManager.get_video_files(input_videos_dir)
    if not video_files:
        print(f"Видеофайлы не найдены в директории {input_videos_dir}")
        return

    VideoFileManager.print_video_list(video_files)
    # "csrt": CSRTTracker(), "kcf": KCFTracker(),
    trackers = {"mosse": MOSSETracker()}

    for tracker_name, tracker in trackers.items():
        print(f"\n{'^' * 50}")
        print(f"Обработка с использованием трекера {tracker_name.upper()}")
        print(f"{'V' * 50}")

        output_dir = VideoFileManager.create_tracker_output_dir(
            output_base_dir, tracker_name
        )

        for i, video_path in enumerate(video_files, 1):
            print(
                f"\nОбработка видео {i}/{len(video_files)}: {os.path.basename(video_path)}"
            )

            video_info = VideoFileManager.get_video_info(video_path)
            if video_info:
                print(
                    f"Файл: {video_info['filename']}, Размер: {video_info['size_mb']} MB"
                )

            try:
                output_path = VideoFileManager.save_processed_video(
                    tracker, video_path, output_dir, tracker_name
                )
                print(
                    f"Успешно обработано и сохранено в: {os.path.basename(output_path)}"
                )
            except Exception as e:
                print(f"Ошибка при обработке {os.path.basename(video_path)}: {str(e)}")

        print(f"\nЗавершена обработка трекером {tracker_name.upper()}")
        print(f"Результаты сохранены в: {output_dir}")


if __name__ == "__main__":
    main()
