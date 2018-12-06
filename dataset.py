
from video import *
from os_utils import safe_makedirs, del_dirs


class DirectoryDataset:
    def __init__(self, directory, video_type='mp4', fps=None):
        self.fps = fps
        self.video_paths = [os.path.join(directory, name)for name in os.listdir(directory) if video_type in name]

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video = Video(self.video_paths[idx])
        if not video:
            raise RuntimeError('Could not load video at path ' + self.video_paths[idx] + '.')
        return video.load_frames(fps=self.fps, delete=False)


class VLOGDataset:
    def __init__(self, labels=None, download_dir='temp', fps=None, start=None, n=None):
        self.fps = fps
        self.download_dir = download_dir
        safe_makedirs(self.download_dir)

        objects = list(range(len(self.object_labels())))
        if labels is not None:
            objects = [self.object_labels().index(label) for label in labels]

        annotations = np.load('vlog_meta/hand_object.npy')
        indices = [np.where(annotations[:, obj] == 1)[0] for obj in objects]
        indices = np.unique(np.hstack(indices))

        links = open('vlog_meta/youtube_links.txt', 'r')
        self.entries = [link for i, link in enumerate(links) if i in indices]
        links.close()

        self.downloader = YoutubeDownloader('mp4')

        # Keep only entries in range [start, start + n]
        self.start = 0 if not start else start
        self.stop = len(self.entries) - self.start if not n else min(self.start + n, len(self.entries))
        self.entries = self.entries[self.start:self.stop+1]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        url, start, stop = self.entries[idx].split(' ')
        start, stop = int(start), int(stop)

        print('\nDownloading video ' + str(self.start + idx) + '...')
        video = self.downloader.download_url(url, self.download_dir)

        try:
            if not video:
                raise RuntimeError('Could not download video from Youtube.')
            frames = video.load_frames(start=start, stop=stop, fps=self.fps, delete=True)
        except RuntimeError as e:
            print(e)
            frames = []

        return self.start + idx, frames

    @staticmethod
    def object_labels():
        file = open('vlog_meta/hand_object_labels.txt', 'r')
        labels = [line.strip() for line in file.readlines()]
        file.close()
        return labels
