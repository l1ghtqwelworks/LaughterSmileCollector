import os
import typing

import librosa
import scipy.signal as signal
from functools import partial

import numpy as np
import tgt
import torch
from scipy.io import wavfile
from torch.utils.data import DataLoader
from tqdm import tqdm

import configs
from models import ResNetBigger
from utils import torch_utils, data_loaders, audio_utils

config = configs.CONFIG_MAP['resnet_with_augmentation']


def load_model_and_device(model_path: str = 'checkpoints/in_use/resnet_with_augmentation') -> (
        ResNetBigger, torch.device):
    model = config['model'](dropout_rate=0.0, linear_layer_size=config['linear_layer_size'],
                            filter_sizes=config['filter_sizes'])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.set_device(device)

    if os.path.exists(model_path):
        torch_utils.load_checkpoint(model_path + '/best.pth.tar', model)
        model.eval()
    else:
        raise Exception(f"Model checkpoint not found at {model_path}")

    print(f"Set device {device}")

    return model, device


def load_audio_dataloader(audio_path: str, sample_rate: int = 8000) -> DataLoader:

    inference_dataset = data_loaders.SwitchBoardLaughterInferenceDataset(
        audio_path=audio_path, feature_fn=config['feature_fn'], sr=sample_rate)

    collate_fn = partial(audio_utils.pad_sequences_with_labels,
                         expand_channel_dim=config['expand_channel_dim'])

    inference_generator = DataLoader(
        inference_dataset, num_workers=4, batch_size=8, shuffle=False, collate_fn=collate_fn)

    print(f"Loaded file: {audio_path}")

    return inference_generator


def lowpass(sig, filter_order=2, cutoff=0.01):
    # Set up Butterworth filter
    B, A = signal.butter(filter_order, cutoff, output='ba')

    # Apply the filter
    return signal.filtfilt(B, A, sig)


def collapse_to_start_and_end_frame(instance_list):
    return instance_list[0], instance_list[-1]


def frame_span_to_time_span(frame_span, fps=100.):
    return frame_span[0] / fps, frame_span[1] / fps


def get_laughter_instances(probs, threshold=0.5, min_length=0.2, fps=100.) -> typing.List[typing.Tuple[float, float]]:
    instances = []
    current_list = []
    for i in range(len(probs)):
        if np.min(probs[i:i + 1]) > threshold:
            current_list.append(i)
        else:
            if len(current_list) > 0:
                instances.append(current_list)
                current_list = []
    if len(current_list) > 0:
        instances.append(current_list)
    instances = [frame_span_to_time_span(collapse_to_start_and_end_frame(i), fps=fps) for i in instances]
    instances = [inst for inst in instances if inst[1] - inst[0] > min_length]
    return instances


def segment_laughter(model: ResNetBigger, device: torch.device, inference_generator: DataLoader, file_length: float,
                     threshold=0.5, min_length=0.2) -> typing.List[typing.Tuple[float, float]]:

    print("Segmenting laughter...")
    
    probs = []
    for model_inputs, _ in tqdm(inference_generator):
        x = torch.from_numpy(model_inputs).float().to(device)
        preds = model(x).cpu().detach().numpy().squeeze()
        if len(preds.shape) == 0:
            preds = [float(preds)]
        else:
            preds = list(preds)
        probs += preds
    probs = np.array(probs)

    fps = len(probs) / float(file_length)

    probs = lowpass(probs)
    instances = get_laughter_instances(probs, threshold=threshold, min_length=float(min_length), fps=fps)

    return instances


def seconds_to_samples(s, sr):
    return s * sr


def cut_laughter_segments(instance_list, y, sr):
    new_audio = []
    for start, end in instance_list:
        sample_start = int(seconds_to_samples(start, sr))
        sample_end = int(seconds_to_samples(end, sr))
        clip = y[sample_start:sample_end]
        new_audio = np.concatenate([new_audio, clip])
    return new_audio


def save_laughter_segments(instances, original_file_path: str, output_dir: str,
                           save_to_audio_files: bool = True,
                           save_to_textgrid: bool = False):
    if len(instances) > 0:
        full_res_y, full_res_sr = librosa.load(original_file_path, sr=44100)
        wav_paths = []
        maxv = np.iinfo(np.int16).max

        if save_to_audio_files:
            if output_dir is None:
                raise Exception("Need to specify an output directory to save audio files")
            else:
                os.system(f"mkdir -p {output_dir}")
                for index, instance in enumerate(instances):
                    laughs = cut_laughter_segments([instance], full_res_y, full_res_sr)
                    wav_path = output_dir + "/laugh_" + str(index) + ".wav"
                    wavfile.write(wav_path, full_res_sr, (laughs * maxv).astype(np.int16))
                    wav_paths.append(wav_path)

        if save_to_textgrid:
            laughs = [{'start': i[0], 'end': i[1]} for i in instances]
            tg = tgt.TextGrid()
            laughs_tier = tgt.IntervalTier(name='laughter', objects=[
                tgt.Interval(l['start'], l['end'], 'laugh') for l in laughs])
            tg.add_tier(laughs_tier)
            fname = os.path.splitext(os.path.basename(original_file_path))[0]
            tgt.write_to_file(tg, os.path.join(output_dir, fname + '_laughter.TextGrid'))

            print('Saved laughter segments in {}'.format(
                os.path.join(output_dir, fname + '_laughter.TextGrid')))


def segment_laughter_file(model: ResNetBigger, device: torch.device, audio_path: str,
                          threshold=0.5, min_length=0.2) -> typing.List[typing.Tuple[float, float]]:
    inference_generator = load_audio_dataloader(audio_path)
    file_length = audio_utils.get_audio_length(audio_path)
    return segment_laughter(model, device, inference_generator, file_length, threshold, min_length)


def load_and_segment_laughter(audio_path: str, threshold=0.5, min_length=0.2) -> typing.List[typing.Tuple[float, float]]:
    model, device = load_model_and_device()
    return segment_laughter_file(model, device, audio_path, threshold, min_length)
