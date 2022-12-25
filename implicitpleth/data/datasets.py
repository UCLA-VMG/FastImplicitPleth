import os
import numpy as np
import imageio.v2 as iio2

class VideoGridDataset(object):
	def __init__(self, video_path, num_frames=900, img_str='rgbd_rgb_', ext='.png', verbose=True):
		""" Video Dataset

		Args:
			img_name (_type_): _description_
		"""
		self.verbose = verbose
		self.video_path = video_path
		self.img_str = img_str
		self.ext = ext
		self.num_frames = num_frames

		self.verbose: print(f'Reading 300 frames from {self.video_path}')
		if os.path.isfile(self.video_path):
			self.vid = iio2.mimread(self.video_path)[:self.num_frames]
			self.vid = np.transpose(self.vid, (1,2,0,3)) # R C T Ch
		elif os.path.isdir(self.video_path):
			self.vid = []
			for frame_num in range(self.num_frames):
				self.vid.append(iio2.imread(os.path.join(self.video_path, self.img_str+str(frame_num)+self.ext)))
			self.vid = np.stack(self.vid, axis=2) # R C T Ch
		else:
			raise NotImplementedError
		self.shape = self.vid.shape
		if self.vid.ndim == 3:
			self.verbose: print('Grayscale. Converting to RGB')
			self.vid = self.vid[...,np.newaxis]
			self.vid = np.concatenate([self.vid,self.vid,self.vid], axis=-1)

		if self.verbose: print(f'Shape of the Video: {self.shape}')
		X, Y, T  = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]), np.arange(self.shape[2]))
		if self.verbose: print(f'Grid Shape -> X: {X.shape}, Y: {Y.shape}, T: {T.shape}')
		self.x = X.ravel()
		self.y = Y.ravel()
		self.t = T.ravel()

		assert len(self.x)*3 == np.prod(self.shape)
		assert len(self.y)*3 == np.prod(self.shape)
		assert len(self.t)*3 == np.prod(self.shape)
		self.num_pixels = len(self.x)

	def __len__(self):
		return self.num_pixels

	def __getitem__(self, idx):
		item_x = (self.x[idx] / self.shape[1]) - 0.5
		item_y = (self.y[idx] / self.shape[0]) - 0.5
		item_t = (self.t[idx] / self.shape[2]) - 0.5
		item_loc = np.array([item_x,item_y,item_t])
		item_rgb = self.vid[self.y[idx],self.x[idx],self.t[idx]]
		return {'pixel': item_rgb, 'loc': item_loc}