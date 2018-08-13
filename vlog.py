
from video import *


class VLOGDataset:
    def __init__(self, labels=None):
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
        video = self.downloader.download_url(url)
        if video is not None:
            video = VideoTransformer.trim(video, int(start), int(stop))
            frames = video.load_frames()
        return frames

    @staticmethod
    def object_labels():
        file = open('meta/hand_object/hand_object_labels.txt', 'r')
        labels = [line.strip() for line in file.readlines()]
        file.close()
        return labels


vlog = VLOGDataset()
