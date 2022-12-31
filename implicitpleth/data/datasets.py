import os
import numpy as np
import imageio.v2 as iio2
import torch

class VideoGridDataset(object):
	def __init__(self, video_path, num_frames=900, start_frame=0, pixel_norm=255, img_str='rgbd_rgb_', ext='.png', verbose=True):
		self.verbose = verbose
		self.video_path = video_path
		self.img_str = img_str
		self.ext = ext
		self.start_frame = start_frame
		self.num_frames = num_frames
		self.pixel_norm = pixel_norm

		self.verbose: print(f'Reading {num_frames} frames from {self.video_path}')
		if os.path.isfile(self.video_path):
			self.vid = iio2.mimread(self.video_path)[self.start_frame:self.start_frame+self.num_frames]
			self.vid = np.transpose(self.vid, (1,2,0,3)) # R C T Ch
		elif os.path.isdir(self.video_path):
			self.vid = []
			for frame_num in range(self.start_frame, self.start_frame+self.num_frames):
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
		x = (torch.tensor(X.ravel()) / self.shape[1]) - 0.5
		y = (torch.tensor(Y.ravel()) / self.shape[0]) - 0.5
		t = (torch.tensor(T.ravel()) / self.shape[2]) - 0.5

		self.vid = torch.tensor(self.vid.reshape(-1,3)) / self.pixel_norm
		self.loc = torch.stack([x,y,t], dim=-1)

		if self.verbose: print(self.vid.shape, self.loc.shape)

		self.num_pixels = len(self.loc)

	def __len__(self):
		return self.num_pixels

	def __getitem__(self, idx):
		return {'pixel': self.vid[idx], 'loc': self.loc[idx]}

	def generate_spatial_tensor_grid(self):
		spatial_X, spatial_Y = np.meshgrid(np.arange(self.shape[1]), np.arange(self.shape[0]))
		spatial_x = (torch.tensor(spatial_X.ravel()) / self.shape[1]) - 0.5
		spatial_y = (torch.tensor(spatial_Y.ravel()) / self.shape[0]) - 0.5
		return torch.stack((spatial_x,spatial_y), dim = -1)
