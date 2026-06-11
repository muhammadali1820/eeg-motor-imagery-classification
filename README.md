# EEG Motor Imagery Classification

A deep learning pipeline to classify brain signals 
(left hand vs right hand motor imagery) using CNN and CNN+LSTM models.

## Dataset
PhysioNet EEG Motor Movement/Imagery Dataset
- 109 subjects, 64 channels, 160 Hz
- 7173 total samples

## Pipeline
1. EEG signal loading with MNE-Python
2. Bandpass filtering (8-30 Hz)
3. Spectrogram conversion (signal to image)
4. CNN baseline model
5. CNN + LSTM hybrid model

## Results
| Model        | Test Accuracy | F1-Score |
|-------------|---------------|----------|
| CNN Baseline | 73.2%        | 0.73     |
| CNN + LSTM   | 76.6%        | 0.76     |

## Tech Stack
Python, MNE-Python, PyTorch, Google Colab

## Results Visualization
![Accuracy](results/accuracy_comparison.png)
![Confusion Matrix](results/confusion_matrix.png)
