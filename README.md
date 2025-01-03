# ABAT
This repository is the implementation of alignment-based adversarial training from paper "Alignment-Based Adversarial Training (ABAT) for Improving the Robustness and Accuracy of EEG-Based BCIs". Please contact xqchen914@hust.edu.cn if you have any questions.


## Data preparation
Downlowd data, and then run MIget2014001.py, MIgetweibo.py, epfldata.py, erp2014009.py, mi2.py to get preprocessed and aligned datasets.

## Run offline cross-session/-block experiments
```python
python main.py --model=EEGNet --dataset=MI2014001 --setup=within --ea=sess --train=ATchastd --AT_eps=0.01
```

## Test cross-session model performance in online scenario
```python
python eval_online.py --model=EEGNet --dataset=MI2014001 --setup=within --ea=sess --train=ATchastd --AT_eps=0.01
```

## Run offline cross-subject domain adaptation experiments
First, pretrain a model using other subjects' data:
```python
python main.py --model=EEGNet --dataset=MI2014001 --setup=cross --ea=sess --train=NT
```
Then, re-train the model on the target subject's data:
```python
python main.py --model=EEGNet --dataset=MI2014001 --setup=within --ea=sess --train=ATchastd --AT_eps=0.01 --FT=1
```

## Run with models with defined number of convolution kernals 
```python
python eval_modelcapa.py --model=ShallowCNN --dataset=MI2014001 --midDim=40 --setup=within --ea=sess --train=ATchastd --AT_eps=0.01
```
```python
python eval_modelcapa.py --model=DeepCNN --dataset=weibo --d1=10 --d2=20 --d3=40 --setup=within --ea=sess --train=ATchastd --AT_eps=0.01
```
