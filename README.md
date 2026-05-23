# DFP-AD
DFP-AD: A Dynamic Frequency Prototypes Framework for Unsupervised Anomaly Detection

# Install Environments
Create a new conda environment and install required packages.

```bash
conda create -n DFP-AD python=3.8.12
conda activate DFP-AD
pip install -r requirements.txt
```

# Multi-Class Setting Train/Test

Train
```bash
python DFP-AD_Multi_Class.py --dataset datasets --data_path .../datsest --phase train
```

Test
```bash
python DFP-AD_Multi_Class.py --dataset datasets --data_path .../datsest --phase test
```

# Few-Shot Setting Train/Test

Train
```bash
python DFP-AD_Few_Shot.py --dataset datasets --data_path .../datsest --shot shot_number --phase train
```

Test
```bash
python DFP-AD_Few_Shot.py --dataset datasets --data_path .../datsest --shot shot_number --phase test
```
# Single-Class Setting Train/Test

Train
```bash
python DFP-AD_Single_Class.py --dataset datasets --data_path .../datsest --phase train
```

Test
```bash
python DFP-AD_Single_Class.py --dataset datasets --data_path .../datsest --phase test
```
# Checkpoints
We provide weights for testing, which you can download from [here](https://pan.baidu.com/s/1ZLc4yrBVh1Fn_71UHa3hog?pwd=7530).

# Acknowledgment
This project benefits from the publicly available implementations of [INP-Former](https://github.com/luow23/INP-Former) and [Dinomaly](https://github.com/guojiajeremy/Dinomaly). We sincerely thank the authors for their valuable contributions to the community.
