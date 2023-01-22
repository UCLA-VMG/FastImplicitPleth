import tqdm
import numpy as np
import imageio.v2 as iio2
import matplotlib.pyplot as plt
import scipy.signal
import torch

class Dict2Class(object):
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])

def trace_video(model, dataset, dataloader, device, plot=True, 
                save_dir=None, save_file="epoch_", save_ext=".avi", verbose=True): 
    model.eval()
    if verbose: print('Tracing Video Grid Points')
    with torch.no_grad():
        temp = []
        for item in dataloader:
            inp = item['loc'].half().to(device)
            output = model(inp) 
            if type(output) == tuple:
                output = output[0] 
            temp.append(output.squeeze().cpu().detach().float().numpy())
    if verbose: print('Arranging Tensor')
    temp = np.concatenate(temp,axis=0).reshape(dataset.shape[0],dataset.shape[1],dataset.shape[2],3)
    temp = np.clip(temp, a_min=0, a_max=1)
    if plot:
        if verbose: print('Plotting 5 frames')
        f, axarr = plt.subplots(1,5, figsize=(20,20))
        plot_frame_num = np.arange(0, dataset.num_frames, dataset.num_frames//5)
        axarr[0].imshow(temp[:,:,plot_frame_num[0]])
        axarr[0].set_title(f'Frame: {plot_frame_num[0]}')
        axarr[0].axis('off')
        axarr[1].imshow(temp[:,:,plot_frame_num[1]])
        axarr[1].set_title(f'Frame: {plot_frame_num[1]}')
        axarr[1].axis('off')
        axarr[2].imshow(temp[:,:,plot_frame_num[2]])
        axarr[2].set_title(f'Frame: {plot_frame_num[2]}')
        axarr[2].axis('off')
        axarr[3].imshow(temp[:,:,plot_frame_num[3]])
        axarr[3].set_title(f'Frame: {plot_frame_num[3]}')
        axarr[3].axis('off')
        axarr[4].imshow(temp[:,:,plot_frame_num[4]])
        axarr[4].set_title(f'Frame: {plot_frame_num[4]}')
        axarr[4].axis('off')
        plt.show()
    if save_dir is not None:
        save_path = f"{save_dir}/{save_file}{save_ext}"
        if verbose: print('Saving Traced Video Data')
        temp = np.transpose(temp*255, (2,0,1,3)).astype(np.uint8)
        iio2.mimwrite(save_path, temp, fps=30)
        if verbose: print(f'Video Saved to {save_path}')
    return temp
    
def trace_video_tqdm(model, dataset, dataloader, device, plot=True, 
                     save_dir=None, save_file="epoch_", save_ext=".avi", verbose=True):
    model.eval()
    if verbose: print('Tracing Video Grid Points')
    with torch.no_grad():
        temp = []
        for item in tqdm(dataloader, leave=False):
            inp = item['loc'].half().to(device)
            temp.append(model(inp)[0].squeeze().cpu().detach().float().numpy())
    if verbose: print('Arranging Tensor')
    temp = np.concatenate(temp,axis=0).reshape(dataset.shape[0],dataset.shape[1],dataset.shape[2],3)
    temp = np.clip(temp, a_min=0, a_max=1)
    if plot:
        if verbose: print('Plotting 5 frames')
        f, axarr = plt.subplots(1,5, figsize=(20,20))
        plot_frame_num = np.arange(0, dataset.num_frames, dataset.num_frames//5)
        axarr[0].imshow(temp[:,:,plot_frame_num[0]])
        axarr[0].set_title(f'Frame: {plot_frame_num[0]}')
        axarr[0].axis('off')
        axarr[1].imshow(temp[:,:,plot_frame_num[1]])
        axarr[1].set_title(f'Frame: {plot_frame_num[1]}')
        axarr[1].axis('off')
        axarr[2].imshow(temp[:,:,plot_frame_num[2]])
        axarr[2].set_title(f'Frame: {plot_frame_num[2]}')
        axarr[2].axis('off')
        axarr[3].imshow(temp[:,:,plot_frame_num[3]])
        axarr[3].set_title(f'Frame: {plot_frame_num[3]}')
        axarr[3].axis('off')
        axarr[4].imshow(temp[:,:,plot_frame_num[4]])
        axarr[4].set_title(f'Frame: {plot_frame_num[4]}')
        axarr[4].axis('off')
        plt.show()
    if save_dir is not None:
        save_path = f"{save_dir}/{save_file}{save_ext}"
        if verbose: print('Saving Traced Video Data')
        temp = np.transpose(temp*255, (2,0,1,3)).astype(np.uint8)
        iio2.mimwrite(save_path, temp, fps=30)
        if verbose: print(f'Video Saved to {save_path}')
    return temp

def positional_encoding(inp, L_max = 10, L_min = 0):
    noDims = inp.shape[-1]
    ctz = 0
    for l in range(L_min,L_max):
        val = 2**l
        for dd in range(noDims):
            if ctz == 0:
                p_enc_grid = torch.cat((torch.sin(np.pi*val*inp[...,dd:dd+1]),torch.cos(np.pi*val*inp[...,dd:dd+1])),dim=-1)
                ctz = 1
            else:
                p_enc_grid = torch.cat((p_enc_grid,torch.sin(np.pi*val*inp[...,dd:dd+1]),torch.cos(np.pi*val*inp[...,dd:dd+1])),dim=-1)
    return p_enc_grid


def positional_encoding_phase(inp, phase, L_max = 10, L_min = 0):
    noDims = inp.shape[-1]
    ctz = 0
    for l in range(L_min,L_max):
        val = 2**l
        base_ix = noDims*(l-L_min)
        for dd in range(noDims):
            if ctz == 0:
                p_enc_grid = torch.cat((torch.sin(np.pi*val*inp[...,dd:dd+1]+phase[...,base_ix+dd:base_ix+dd+1]),torch.cos(np.pi*val*inp[...,dd:dd+1]+phase[...,base_ix+dd:base_ix+dd+1])),dim=-1)
                ctz = 1
            else:
                p_enc_grid = torch.cat((p_enc_grid,torch.sin(np.pi*val*inp[...,dd:dd+1]+phase[...,base_ix+dd:base_ix+dd+1]),torch.cos(np.pi*val*inp[...,dd:dd+1]+phase[...,base_ix+dd:base_ix+dd+1])),dim=-1)
    return p_enc_grid

def custom_detrend(*args,**kwargs):
    raise NotImplementedError


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
    if DETREND:
        BVP = custom_detrend(np.cumsum(BVP), 100)
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