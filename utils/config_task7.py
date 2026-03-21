import pandas as pd
import os

sample_rate = 32000
clip_samples = sample_rate * 4

mel_bins = 64
fmin = 50
fmax = 14000
window_size = 1024
hop_size = 320
window = 'hann'
pad_mode = 'reflect'
center = True
device = 'cuda'
ref = 1.0
amin = 1e-10
top_db = None
classes_num_DIL = 10


df_DIL_dev_train = pd.read_csv("/scratch/project_462001198/manjunath/DIL/task7_data/evaluation_setup/development_train.txt", sep='\t', names=['filename', 'target', 'domain', 'new_target'])
df_DIL_dev_test = pd.read_csv("/scratch/project_462001198/manjunath/DIL/task7_data/evaluation_setup/development_test.txt", sep='\t', names=['filename', 'target', 'domain', 'new_target'])
#df_DIL_eval = pd.read_csv("/scratch/project_462001198/manjunath/DIL/task7_data/evaluation_setup/development_test.txt", sep='\t', names=['filename', 'target', 'new_target'])
audio_folder_DIL = "/scratch/project_462001198/manjunath/DIL/task7_data/"
