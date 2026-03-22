This repository contains the baseline code for DCASE 2026 Task 7, Domain-Agnostic Incremental Learning for Audio Classification.
Participants can build their own systems by extending the provided baseline system. 

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```
## Data and Features 

* Download the development dataset from the
* The dataset structure should be:
```text
task7_data/
├── audio
└── evaluation_setup ├── development_train.txt
                     └── development_test.txt
```
* Update the path of meta files (development_train.txt and development_test.txt) in the [config_task7](utils/config_task7.py) file.
* Both development_train.txt and development_test.txt include: file-name, class, domain-id and label:
```text
audio/00333.wav	alarm	D2	0
audio/00334.wav	alarm	D2	0
audio/00335.wav	alarm	D2	0
audio/07671.wav	speech	D3	9
audio/07672.wav	speech	D3	9
audio/07673.wav	speech	D3	9
```
* Labels for corresponding sounds are listed in [config_task7](utils/config_task7.py).
* We segmented the audios in development train into 4-second signals for training the baseline system in batches. The testing samples have variable lengths. During inference, we predict the class label per audio file. However, participants can choose any method to deal with audios in variable lengths.
* Log mel-band energies are obtained from sounds using [torchlibrosa](https://github.com/qiuqiangkong/torchlibrosa) library.
## System description
The baseline system includes 6 convolutional blocks. Each block includes 2 convolutional layers, each convolutional layer followed by a batch normalization (BN) layer, with the layer specifications the same as [PANNs](https://github.com/qiuqiangkong/audioset_tagging_cnn) CNN14.  Global pooling is applied to the last convolutional layer, to get a fixed-length input feature vector to the classifier. The baseline model is trained from scratch on the domain D1, then separate domain-specific BN layers are adapted for domain D2 and D3 in incremental phases. 
![](./task7.jpg)
During inference, domain-specific BN layers are predicted and used with domain-shared layers for classification. Specifically, an input audio is forward passed through a combination of shared and domain-specific layers of each domain seen so far and obtains the class probabilities. Subsequently, uncertainty of the model on given input audio among the predicted probabilities is computed using entropy. The domain-specific layers which have minimum entropy, denoting lower uncertainty, are selected for classification.
## Parameters
#### Acoustic features
- Sampling rate:  32 kHz
- Training samples in the development set are segmented into 4-second signals, while the testing samples have variable lengths.
- Log mel-band energies (64 bands) with lower and upper cut-off frequencies 50 Hz and 14 kHz respectively. The window (Hamming) size is set to 1024 samples  and hop size to 320 samples. 
#### Neural network
- Architecture
  - CNN block #1:  
    - 2 x [2D Convolutional layer (filters: 64, kernel size: 3) + Batch normalization + ReLu], 2 x 2 average pooling + Dropout (rate: 20%)
  - CNN block #2:  
    - 2 x [2D Convolutional layer (filters: 128, kernel size: 3) + Batch normalization + ReLu], 2 x 2 average pooling + Dropout (rate: 20%)
  - CNN block #3:  
    - 2 x [2D Convolutional layer (filters: 256, kernel size: 3) + Batch normalization + ReLu], 2 x 2 average pooling + Dropout (rate: 20%)
  - CNN block #4:  
    - 2 x [2D Convolutional layer (filters: 512, kernel size: 3) + Batch normalization + ReLu], 2 x 2 average pooling + Dropout (rate: 20%)
  - CNN block #5:  
    - 2 x [2D Convolutional layer (filters: 1024, kernel size: 3) + Batch normalization + ReLu], 2 x 2 average pooling + Dropout (rate: 20%)
  - CNN block #6:  
    - 2 x [2D Convolutional layer (filters: 2048, kernel size: 3) + Batch normalization + ReLu], 2 x 2 average pooling + Dropout (rate: 20%)
  - Global pooling
  - Output layer (activation: softmax)
- Learning: 120 epochs (batch size 32), data shuffling between epochs
- Optimizer: Adam (learning rate at initial phase: 0.0001, at incremental phases: 0.00001)
- Scheduler: CosineAnnealingLR

## Results for the development dataset:

Results of baseline are calculated using PyTorch in GPU mode . The baseline is trained for 120 epochs and tested on the test split of the development dataset.
<table class="dataset-table">
  <thead>
    <tr>
      <th></th>
      <th>D2</th>
      <th>D3</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>D3</td>
      <td></td>
      <td>33.9</td>
    </tr>
    <tr>
      <td>D2</td>
      <td>54.3</td>
      <td>53.9</td>
    </tr>
    <tr>
      <td>Average accuracy</td>
      <td>54.3</td>
      <td>43.9</td>
    </tr>
    
  </tbody>
</table>
Note: The reported baseline system performance is not exactly reproducible due to varying setups. However, you should be able to obtain very similar results.
## Citation
If you are using the baseline system, please cite the following: 

```BibTeX
@INPROCEEDINGS{10890481,
  author={Manjunath Mulimani and Annamaria Mesaros},
  booktitle={IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={Domain-Incremental Learning for Audio Classification}, 
  year={2025}, 
  pages={1-5}
  }
```