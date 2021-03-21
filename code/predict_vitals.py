import tensorflow as tf
import numpy as np
import scipy.io
import os
import sys
import argparse
sys.path.append('../')
from model import Attention_mask, MTTS_CAN
import h5py
import matplotlib.pyplot as plt
from scipy.signal import butter
from inference_preprocess import preprocess_raw_video, detrend
import librosa
import librosa.display
from pathlib import Path
from post_process import calculate_metric

def predict_vitals(args):

    print(Path(os.path.basename(args.video_path)).stem)

    fs = args.sampling_rate

    preds_file_name = r'preds/preds_{}.npy'.format(Path(os.path.basename(args.video_path)).stem)

    if os.path.exists(preds_file_name):
        pulse_pred, resp_pred = np.load(preds_file_name)
    else:
        img_rows = 36
        img_cols = 36
        frame_depth = 10
        model_checkpoint = '../mtts_can.hdf5'
        batch_size = args.batch_size
        fs = args.sampling_rate
        sample_data_path = args.video_path

        dXsub = preprocess_raw_video(sample_data_path, dim=36)
        # print('dXsub shape', dXsub.shape)

        dXsub_len = (dXsub.shape[0] // frame_depth)  * frame_depth
        dXsub = dXsub[:dXsub_len, :, :, :]

        model = MTTS_CAN(frame_depth, 32, 64, (img_rows, img_cols, 3))
        model.load_weights(model_checkpoint)

        yptest = model.predict((dXsub[:, :, :, :3], dXsub[:, :, :, -3:]), batch_size=batch_size, verbose=1)

        pulse_pred = yptest[0]
        resp_pred = yptest[1]

        os.makedirs('preds', exist_ok=True)
        np.save(preds_file_name, [pulse_pred, resp_pred])

    MAE, RMSE, meanSNR, HR0, HR = calculate_metric(pulse_pred, labels=[], signal='pulse', window_size=360, fs=30,
                                                   bpFlag=True)

    pulse_pred = detrend(np.cumsum(pulse_pred), 100)
    [b_pulse, a_pulse] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    pulse_pred = scipy.signal.filtfilt(b_pulse, a_pulse, np.double(pulse_pred))

    resp_pred = detrend(np.cumsum(resp_pred), 100)
    [b_resp, a_resp] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    resp_pred = scipy.signal.filtfilt(b_resp, a_resp, np.double(resp_pred))

    pulse_fft = np.abs(np.fft.rfft(pulse_pred))
    resp_fft = np.abs(np.fft.rfft(resp_pred))
    freq = np.fft.rfftfreq(len(pulse_pred), 1/args.sampling_rate)

    pulse_max_freq = freq[np.argmax(pulse_fft)]
    resp_max_freq = freq[np.argmax(resp_fft)]

    print("Mean SNR: {:.2f}".format(meanSNR))
    print("Heart Rate: {} beats per min".format(round(pulse_max_freq * 60)))
    print("Resp Rate: {} breaths per min".format(round(resp_max_freq * 60)))

    if args.plot:
        ########## Plot ##################
        t = np.arange(0,len(pulse_pred)/args.sampling_rate,1/args.sampling_rate)[:len(pulse_pred)]
        #Pulse Signal + FFT
        for i, (sig, fft, max_freq) in enumerate([(pulse_pred, pulse_fft, pulse_max_freq), (resp_pred, resp_fft,resp_max_freq)]):
            typ = "Heart" if i == 0 else "Resp"
            unit = "beats" if i == 0 else "breaths"
            plt.figure()
            plt.subplot(121)
            plt.plot(t,sig/max(abs(sig)))
            plt.title('Prediction of {} Rate Signal'.format(typ))
            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.subplot(122)
            plt.plot(freq * 60, fft)
            plt.xlabel("Frequency (hz)")
            plt.ylabel("Amplitude")
            plt.title("{} Rate: {} {} per min".format(typ, round(max_freq * 60),unit))

        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, help='processed video path')
    parser.add_argument('--sampling_rate', type=int, default = 30, help='sampling rate of your video')
    parser.add_argument('--batch_size', type=int, default = 100, help='batch size (multiplier of 10)')
    parser.add_argument('--plot', default=False, action='store_true', help='plots signal and fft')
    args = parser.parse_args()

    ### SINGLE VIDEO
    videoFilePathFolder = 'videos'
    videoFilePathFile = 'matt_2.mp4'
    args.video_path = os.path.join(videoFilePathFolder, videoFilePathFile)
    args.plot = True#False
    predict_vitals(args)

    #MULTIPLE VIDEOS
    # args.plot = False
    # videoFilePathFolder = r'videos/'
    # for f in sorted([f for f in os.listdir(videoFilePathFolder) if f.endswith(('.mp4'))]):
    #     args.video_path = os.path.join(videoFilePathFolder,f)
    #     predict_vitals(args)






