import os
import re
import zipfile
import shutil
from PIL import Image
import tensorflow as tf
from tensorflow_datasets.core.utils import py_utils
import json


# TODO add tar.gz support
# TODO check types with python-magic

class Holder:

	def __init__(self, name, file_type, path, output_path=None):
		self.name = name  # image
		self.typ = file_type  # png
		self.path = path  # /folder/image.png
		self.output_path = output_path  # /target_folder/image.png


class ImageHolder(Holder):

	def __init__(self, zip_file=None, *args, **kwargs):
		super(ImageHolder, self).__init__(*args, **kwargs)
		self.zip_file = zip_file

	def image_size(self):
		try:
			if self.zip_file:
				infile = self.path
				file = self.zip_file.open(infile, 'r')
			else:
				file = self.path
			im = Image.open(file)
			return im.size
		except OSError:
			return 10, 10

	def create_fakes(self):
		basedir = os.path.dirname(self.output_path)
		if not tf.io.gfile.exists(basedir):
			tf.io.gfile.makedirs(basedir)
		print('created, ', self.output_path, ', size: ', self.image_size())
		img = Image.new('RGB', self.image_size(), (255, 255, 255))
		img.save(self.output_path)


class PlainTextHolder(Holder):

	def __init__(self, *args, **kwargs):
		super(PlainTextHolder, self).__init__(*args, **kwargs)

	def create_fakes(self):
		out = tf.io.gfile.GFile(self.output_path, mode='w')
		with tf.io.gfile.GFile(self.path, mode='r') as inf:
			count = 0
			breaker = 0
			while count < 5 and breaker < 30:  # write 5 non empty line
				line = inf.readline()
				out.write(line)
				print(line)
				if not line.rstrip():
					count += 1
				breaker += 1

		out.close()


class ZipHolder(Holder):
	def __init__(self, *args, **kwargs):
		super(ZipHolder, self).__init__(*args, **kwargs)

	def create_fakes(self):
		zip_file = zipfile.ZipFile(self.path)
		f = zip_file.namelist()
		r = re.compile(".*/$")
		folders = list(filter(r.match, f))  # it's catch the folders names
		ex_files = []
		print(folders)
		for prefix in folders:  # take 2 example from the folders
			ex_files += list(filter(lambda x: x.startswith(prefix), f))[1:3]

		for file in ex_files:
			name = os.path.basename(file)
			typ = os.path.splitext(file)[1]
			target_path = os.path.join(os.path.splitext(self.output_path)[0], file)
			hold = HolderFactory(zip_file, name, typ, file,
													 target_path).generate_holder()
			hold.create_fakes()

		zip_file.close()
		folder_path = os.path.join(os.path.splitext(self.output_path)[0])
		shutil.make_archive(os.path.splitext(self.output_path)[0], 'zip',
												folder_path)
		tf.io.gfile.rmtree(folder_path)  # delete created unzipped folder


class HolderFactory(Holder):
	def __init__(self, zip_file=None, *args, **kwargs):
		super(HolderFactory, self).__init__(*args, **kwargs)
		self.zip_file = zip_file

	def generate_holder(self):

		if self.path.endswith('.zip'):
			return ZipHolder(self.name, self.typ, self.path, self.output_path)
		elif self.path.endswith(('.jpg', '.jpeg', '.png', '.tiff')):
			return ImageHolder(self.zip_file, self.name, self.typ, self.path,
												 self.output_path)
		elif self.path.endswith(
				('.csv', '.txt', '.en', '.ne', '.si', '.data', '.md')):
			return PlainTextHolder(self.name, self.typ, self.path, self.output_path)


class Generator:
	def __init__(self, dataset_name):
		self.dataset_name = dataset_name
		self.inpath = self.dataset_folder_finder()
		self.outpath = os.path.join(os.path.join(py_utils.tfds_dir(), 'testing',
																						 'test_data', 'fake_examples',
																						 os.path.basename(
																							 self.inpath) + 'auto_gen'))

	def dataset_folder_finder(self):
		home = os.path.expanduser('~')
		path = os.path.join(home, 'tensorflow_datasets', 'downloads')

		for r, d, f in os.walk(path):
			for file in f:
				if ".INFO" in file:
					aha = os.path.join(r, file)
					filename = os.path.splitext(aha)[0]
					with open(aha) as data_file:
						data_item = json.load(data_file)
						if data_item['dataset_names'][0] == self.dataset_name:
							return filename
		# raise error
		raise FileNotFoundError(
			'Dataset not found in `{}`. Please be sure the dataset is downloaded!'.format(
				path))

	def generator(self):
		if self.inpath.endswith('.zip'):
			self.zip_generator()
		else:
			# eger direk zip file gelirse onune bi checker koy zipfile a gonder dosyayi yaratip
			for dirpath, dirnames, filenames in tf.io.gfile.walk(self.inpath):
				structure = os.path.join(self.outpath,
																 os.path.relpath(dirpath, self.inpath))
				if not tf.io.gfile.isdir(structure):
					tf.io.gfile.mkdir(structure)
				else:
					print("Folder does already exits!")
				count = 0
				while count < 2:  # take just 2 files on one folder
					try:
						file = filenames[count]
					except IndexError:
						break
					file_path = os.path.join(dirpath, file)
					file_target_path = os.path.join(structure, file)
					name = os.path.basename(file_path)
					typ = os.path.splitext(file_path)[1]
					hold = HolderFactory(None, name, typ, file_path, file_target_path)
					try:
						hold.generate_holder().create_fakes()
					except AttributeError:
						pass

					count += 1