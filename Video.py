
import cv2
from pytube import YouTube
import os


class Video:
    def __init__(self, file_name):
        self.name = file_name
        self.basename = os.path.split(self.name)[-1].split('.')[0]

        cap = cv2.VideoCapture(file_name)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def load_frames(self):
        cap = cv2.VideoCapture(self.name)
        frames = []
        for _ in range(self.n_frames):
            success, frame = cap.read()
            if not success:
                break
            frames.append(frame)
        cap.release()
        return frames


class YoutubeDownloader:
    def __init__(self, file_type='mp4'):
        self.file_type = file_type
        self.video_id = 0

    def download_url(self, url, directory):
        filename = str(self.video_id)
        self.video_id += 1

        yt = YouTube(url)
        yt.streams.filter(progressive=True, file_extension=self.file_type) \
            .order_by('resolution') \
            .desc() \
            .first() \
            .download(output_path=directory, filename=filename)
        return Video(os.path.join(directory, filename + '.' + self.file_type))


class VideoTransformer:
    def trim(self, video, start_frame, stop_frame, directory):
        cap = cv2.VideoCapture(video.name)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        name = os.path.join(directory, video.basename.split('.')[0] + '.avi')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(name, fourcc, video.fps, (video.width, video.height))

        for _ in range(stop_frame - start_frame + 1):
            _, frame = cap.read()
            out.write(frame)
        cap.release()
        out.release()

        return Video(name)
