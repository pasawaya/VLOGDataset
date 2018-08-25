
import cv2
from pytube import YouTube
from pytube.exceptions import RegexMatchError
import os
import numpy as np


class Video:
    def __init__(self, file_name):
        self.name = file_name
        self.basename = os.path.split(self.name)[-1].split('.')[0]

        cap = cv2.VideoCapture(file_name)
        self.n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(cap.get(cv2.CAP_PROP_FPS))
        cap.release()

    def load_frames(self, start, stop, fps=None):
        if fps is None:
            fps = self.fps

        assert fps <= self.fps

        frame_indices = np.linspace(start, stop, num=int(fps * (stop - start) / self.fps), dtype=np.int)
        cap = cv2.VideoCapture(self.name)
        frames = []

        for i in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            success, frame = cap.read()
            if not success:
                raise RuntimeError('Could not read frames.')
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        cap.release()
        return frames


class YoutubeDownloader:
    def __init__(self, file_type='mp4', start_id=0):
        self.file_type = file_type
        self.video_id = start_id

    def download_url(self, url, directory):
        filename = str(self.video_id)
        self.video_id += 1

        try:
            yt = YouTube(url)
        except (RegexMatchError, KeyError):
            return None

        yt.streams.filter(progressive=True, file_extension=self.file_type) \
            .order_by('resolution') \
            .desc() \
            .first() \
            .download(output_path=directory, filename=filename)
        return Video(os.path.join(directory, filename + '.' + self.file_type))


class VideoTransformer:
    @staticmethod
    def write(frames, path, fps):
        (h, w, _) = frames[0].shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(path, fourcc, fps, (w, h))

        for frame in frames:
            if len(frame) == 2:
                frame = np.expand_dims(frame, 2)
                frame = np.repeat(frame, 3, axis=2)
            out.write(frame)
        out.release()

    @staticmethod
    def scale(image, bbox, f_xy):
        (h, w, _) = image.shape

        new_w = int(w * f_xy)
        new_h = int(h * f_xy)

        image = cv2.resize(image, (new_w, new_h))
        bbox = bbox * f_xy

        return image, bbox.astype(np.int32)

    @staticmethod
    def crop(image, bbox, mask, length):
        y1, x1, y2, x2 = bbox
        x_min, y_min, x_max, y_max = min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)
        w_obj, h_obj = abs(x2 - x1), abs(y2 - y1)

        x_obj, y_obj = x_min + (w_obj / 2), y_min + (h_obj / 2)
        (h, w, _) = image.shape

        x_min_new, y_min_new = int(max(0, x_obj - (length / 2))), int(max(0, y_obj - (length / 2)))
        x_max_new, y_max_new = int(min(w - 1, x_min_new + length)), int(min(h - 1, y_min_new + length))

        image = image[y_min_new:y_max_new, x_min_new:x_max_new, :]
        mask = mask[y_min_new:y_max_new, x_min_new:x_max_new]
        return image, mask
