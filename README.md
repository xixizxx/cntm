# CNTM
CNTM is a neural topic model for contrastive disentangled topic learning, based on the word embeddings and topic embeddings learned in the same representation space.

## Dependencies
+ pytorch

## Datasets
All the datasets (20ng/webs/tmn/reuters) are pre-processed by the scripts in the subfolder: preprocess. The argument configs are set by json files under the `configs' folder. Please refer to our paper and download the datasets.

## To train CNTM, please run:
```
python train.py 20ng
python train.py webs
python train.py tmn
python train.py reuters
```

## To evaluate CNTM on topic coherence, topic diversity, and visualize the results by t-SNE, please run:
```
python evaluate.py 20ng
python evaluate.py webs
python evaluate.py tmn
python evaluate.py reuters
```


