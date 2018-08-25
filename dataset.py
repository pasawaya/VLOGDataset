
from video import *


class VLOGDataset:
    def __init__(self, labels=None, download_dir='temp', verbose=True, fps=None):
        self.verbose = verbose
        self.fps = fps
        self.download_dir = download_dir
        if not os.path.exists(self.download_dir):
            os.makedirs(download_dir)

        objects = list(range(len(self.object_labels())))
        if labels is not None:
            objects = [self.object_labels().index(label) for label in labels]

        annotations = np.load('meta/hand_object/hand_object.npy')
        indices = [np.where(annotations[:, obj] == 1)[0] for obj in objects]
        indices = np.unique(np.hstack(indices))

        links = open('meta/youtube_links.txt', 'r')
        self.entries = [link for i, link in enumerate(links) if i in indices]
        links.close()

        self.downloader = YoutubeDownloader('mp4')

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        url, start, stop = self.entries[idx].split(' ')
        start, stop = int(start), int(stop)

        if self.verbose:
            print('Downloading video ' + str(idx) + '...')
        video = self.downloader.download_url(url, self.download_dir)

        if video is None:
            raise RuntimeError('Could not download video from Youtube.')

        if self.verbose:
            print('Trimming video ' + str(idx) + '...')

        return video.load_frames(start, stop, fps=self.fps)

    @staticmethod
    def object_labels():
        file = open('meta/hand_object/hand_object_labels.txt', 'r')
        labels = [line.strip() for line in file.readlines()]
        file.close()
        return labels


vlog = VLOGDataset()
