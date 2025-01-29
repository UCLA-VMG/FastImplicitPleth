
import os
import numpy as np
import imageio.v2 as iio2
import matplotlib.pyplot as plt
import scipy.signal
import torch
from torch import nn
import tinycudann as tcnn

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])


def prpsd2(BVP, FS, LL_PR, UL_PR, BUTTER_ORDER=6, DETREND=False, PlotTF=False, FResBPM = 0.1,RECT=True):
    '''
    Estimates pulse rate from the power spectral density a BVP signal
    
    Inputs
        BVP              : A BVP timeseries. (1d numpy array)
        fs               : The sample rate of the BVP time series (Hz/fps). (int)
        lower_cutoff_bpm : The lower limit for pulse rate (bpm). (int)
        upper_cutoff_bpm : The upper limit for pulse rate (bpm). (int)
        butter_order     : Order of the Butterworth Filter. (int)
        detrend          : Detrend the input signal. (bool)
        FResBPM          : Resolution (bpm) of bins in power spectrum used to determine pulse rate and SNR. (float)
    
    Outputs
        pulse_rate       : The estimated pulse rate in BPM. (float)
    
    Daniel McDuff, Ethan Blackford, January 2019
    Copyright (c)
    Licensed under the MIT License and the RAIL AI License.
    '''

    N = (60*FS)/FResBPM

    # Detrending + nth order butterworth + periodogram
    # if DETREND:
    #     BVP = custom_detrend(np.cumsum(BVP), 100)
    if BUTTER_ORDER:
        [b, a] = scipy.signal.butter(BUTTER_ORDER, [LL_PR/60, UL_PR/60], btype='bandpass', fs = FS)
    
        BVP = scipy.signal.filtfilt(b, a, np.double(BVP))
    
    # Calculate the PSD and the mask for the desired range
    if RECT:
        F, Pxx = scipy.signal.periodogram(x=BVP,  nfft=N, fs=FS, detrend=False);  
    else:
        F, Pxx = scipy.signal.periodogram(x=BVP, window=np.hanning(len(BVP)), nfft=N, fs=FS)
    FMask = (F >= (LL_PR/60)) & (F <= (UL_PR/60))
    
    # Calculate predicted pulse rate:
    FRange = F * FMask
    PRange = Pxx * FMask
    MaxInd = np.argmax(PRange)
    pulse_rate_freq = FRange[MaxInd]
    pulse_rate = pulse_rate_freq*60

    # Optionally Plot the PSD and peak frequency
    if PlotTF:
        # Plot PSD (in dB) and peak frequency
        plt.figure()
        plt.plot(F, 10 * np.log10(Pxx))
        plt.plot(pulse_rate_freq, 10 * np.log10(PRange[MaxInd]),'ro')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power (dB)')
        plt.xlim([0, 4.5])
        plt.title('Power Spectrum and Peak Frequency')
            
    return pulse_rate

def getErrors(bpmES, bpmGT, timesES=None, timesGT=None):
    RMSE = RMSEerror(bpmES, bpmGT, timesES, timesGT)
    MAE = MAEerror(bpmES, bpmGT, timesES, timesGT)
    MAX = MAXError(bpmES, bpmGT, timesES, timesGT)
    PCC = PearsonCorr(bpmES, bpmGT, timesES, timesGT)
    return RMSE, MAE, MAX, PCC

def RMSEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ RMSE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.zeros(n)
    for j in range(m):
        for c in range(n):
            df[c] += np.power(diff[c,j],2)

    # -- final RMSE
    RMSE = np.sqrt(df/m)
    return RMSE

def MAEerror(bpmES, bpmGT, timesES=None, timesGT=None):
    """ MAE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.sum(np.abs(diff),axis=1)

    # -- final MAE
    MAE = df/m
    return MAE

def MAXError(bpmES, bpmGT, timesES=None, timesGT=None):
    """ MAE: """

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    df = np.max(np.abs(diff),axis=1)

    # -- final MAE
    MAX = df
    return MAX

def PearsonCorr(bpmES, bpmGT, timesES=None, timesGT=None):
    from scipy import stats

    diff = bpm_diff(bpmES, bpmGT, timesES, timesGT)
    n,m = diff.shape  # n = num channels, m = bpm length
    CC = np.zeros(n)
    for c in range(n):
        # -- corr
        r,p = stats.pearsonr(diff[c,:]+bpmES[c,:],bpmES[c,:])
        CC[c] = r
    return CC

def bpm_diff(bpmES, bpmGT, timesES=None, timesGT=None):
    n,m = bpmES.shape  # n = num channels, m = bpm length

    if (timesES is None) or (timesGT is None):
        timesES = np.arange(m)
        timesGT = timesES
            
    diff = np.zeros((n,m))
    for j in range(m):
        t = timesES[j]
        i = np.argmin(np.abs(t-timesGT))
        for c in range(n):
            diff[c,j] = bpmGT[i]-bpmES[c,j]
    return diff


class  AppearanceNet(torch.nn.Module):
    def __init__(self, spatiotemporal_to_delta_encoding, spatiotemporal_to_delta_network, 
                 deltaspatial_to_rgb_encoding, deltaspatial_to_rgb_network):
        super().__init__()
        self.xyt_to_d_enc = tcnn.Encoding(spatiotemporal_to_delta_encoding["input_dims"], 
                                          spatiotemporal_to_delta_encoding)
        self.xyt_to_d_net = tcnn.Network(self.xyt_to_d_enc.n_output_dims, spatiotemporal_to_delta_network["output_dims"], 
                                         spatiotemporal_to_delta_network)

        self.d_to_rgb_enc = tcnn.Encoding(deltaspatial_to_rgb_encoding["input_dims"], 
                                          deltaspatial_to_rgb_encoding)
        self.d_to_rgb_net = tcnn.Network(self.d_to_rgb_enc.n_output_dims, deltaspatial_to_rgb_network["output_dims"], 
                                         deltaspatial_to_rgb_network)
        self.spatiotemporal_to_delta = torch.nn.Sequential(self.xyt_to_d_enc, self.xyt_to_d_net)
        self.deltaspatial_to_rgb = torch.nn.Sequential(self.d_to_rgb_enc, self.d_to_rgb_net)
        self.device_spatiotemporal_to_delta = torch.device("cpu")
        self.device_deltaspatial_to_rgb_device = torch.device("cpu")
    
    def forward(self, coords):
        coords = coords.to(self.device_spatiotemporal_to_delta)
        delta = self.spatiotemporal_to_delta(coords)
        interim_out = coords[...,0:2] + (delta/2) # tanh is -1 to 1. But we need -0.5 to 0.5
        out = self.deltaspatial_to_rgb(interim_out.to(self.device_deltaspatial_to_rgb_device))
        return out.to(self.device_spatiotemporal_to_delta), {"interim_out": interim_out, "delta": delta}
    
    def set_device(self,device_spatiotemporal_to_delta, device_deltaspatial_to_rgb):
        self.device_spatiotemporal_to_delta = device_spatiotemporal_to_delta
        self.device_deltaspatial_to_rgb_device = device_deltaspatial_to_rgb
        # Move to device
        self.spatiotemporal_to_delta.to(self.device_spatiotemporal_to_delta)
        self.deltaspatial_to_rgb.to(self.device_deltaspatial_to_rgb_device)

class CNN3D(nn.Module):
    def __init__(self, frames=64, sidelen = 128, channels=3):  
        super(CNN3D, self).__init__()
        
        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(channels, 16, [1,5,5],stride=1, padding=[0,2,2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
 
        self.ConvBlock10 = nn.Conv3d(64, 1, [1,1,1],stride=1, padding=0)
       
        self.MaxpoolTem = nn.MaxPool3d((2, 1, 1), stride=(2, 1, 1))
       
        self.poolspa = nn.AdaptiveAvgPool3d((1,sidelen,sidelen))

        
    def forward(self, x):
        x = x.permute(0,4,1,2,3)
        [batch,channel,length,width,height] = x.shape
        x = self.ConvBlock1(x)
        x = self.ConvBlock2(x)
        x_visual6464 = self.ConvBlock3(x)
        x = self.MaxpoolTem(x_visual6464)
        x = self.ConvBlock4(x)
        x = self.poolspa(x)   
        x = torch.sigmoid(self.ConvBlock10(x)) 
        return x.reshape(x.shape[0],x.shape[3],x.shape[4])

class VideoGridDataset(object):
	def __init__(self, video_path, num_frames=900, start_frame=0, pixel_norm=255, img_str='rgbd_rgb_', ext='.png', verbose=True, positive_coord=False):
		self.positive_coord = positive_coord
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
			raise FileNotFoundError(f'No such file: {self.video_path}')
		self.shape = self.vid.shape
		if self.vid.ndim == 3:
			self.verbose: print('Grayscale. Converting to RGB')
			self.vid = self.vid[...,np.newaxis]
			self.vid = np.concatenate([self.vid,self.vid,self.vid], axis=-1)

		if self.verbose: print(f'Shape of the Video: {self.shape}')
		if self.positive_coord:
			half_dx =  0.5 / self.shape[1]
			half_dy =  0.5 / self.shape[0]
			half_dt =  0.5 / self.shape[2]
			xs = np.linspace(half_dx, 1-half_dx, self.shape[1])
			ys = np.linspace(half_dy, 1-half_dy, self.shape[0])
			ts = np.linspace(half_dt, 1-half_dt, self.shape[2])
			X, Y, T = np.meshgrid(xs, ys, ts)
			if self.verbose: print(f'Linspace Grid Shape -> X: {X.shape}, Y: {Y.shape}, T: {T.shape}')
			x = torch.tensor(X.ravel())
			y = torch.tensor(Y.ravel())
			t = torch.tensor(T.ravel())
		else:
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