# ABAT
This repository is the implementation of alignment-based adversarial training.

## Data preparation
Run MIget2014001.py, MIgetweibo.py, epfldata.py, erp2014009.py, mi2.py to get preprocessed datasets.

## Run offline cross-session/-block experiments
```python
python main.py --model=EEGNet --dataset=MI2014001 --setup=within --ea=sess --train=ATchastd
```
