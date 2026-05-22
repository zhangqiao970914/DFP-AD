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
