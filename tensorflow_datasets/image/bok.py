#
from tensorflow_datasets.core.download.kaggle import KaggleCompetitionDownloader
# from tensorflow_datasets.core.dataset_builder.


downloader = KaggleCompetitionDownloader('digit-recognizer')
import os
import tensorflow_datasets as tfds
print(os.getcwd())
bokum = os.getcwd()
#downloader.download_file('test.csv', bokum)
print(downloader.competition_files)
# for x, got in enumerate(downloader.competition_files):
#     downloader.download_file(got, bokum)
#     print("---- ", x, got, type(got))


