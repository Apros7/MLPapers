# [Learning A Driving Simulator](https://arxiv.org/pdf/1608.01230): Notes on first Comma AI driving simulator

Some keypoints:
- No world assumptions are made: the simulator is entirely based on training from the dataset
- The focus is Camera frames, steering angle and speed.
- The data is 80 x 160 with values from -1 to 1

## Files:

- train_*.py: files to train a simulator in python using the 7.25 hours of video available from comma using a certain architecture.
- sample.py: samples a run given a (real) starting frame letting the RNN "hallucinate".

