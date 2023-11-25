# SPP
This repository contains an implementation of separation prority pipeline (SPP) for noisy speech separation.

The code is based on the [Asteroid toolkit](https://github.com/asteroid-team/asteroid) for audio speech separation.

## Running the experiments
The directory `egs` contains a recipe for [Libri2mix dataset](https://github.com/JorisCos/LibriMix). The recipe assumes that you already have the LibriMix dataset available. If not, please follow the instructions at the [LibriMix repository](https://github.com/JorisCos/LibriMix) to obtain it. 

cd egs/libri2mix

### Training SPP
To train the SPP run
```
python train.py
```

### Evalating SPP
To evaluate the SPP run
```
python eval.py
```

