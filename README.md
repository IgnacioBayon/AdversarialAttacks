# AdversarialAttacks

## How to train:

The train and evaluation scripts of either model (Sentiment Analysis RNN and News Classification RNN) can be found in their respective packages (src.sentimentAnalysis, src.newsClassification).

To train sentiment analysis model: 
```
'python3 -m src.sentimentAnalysis.train'
```
to train news classification model: 
```
'python3 -m src.newsClassification.train'
```

## How to evaluate

To evaluate, simply run the evaluate script of each package (sentimentAnalysis and newsClassification). To evaluate correctly, the model state dicts must be saved inside a models directory in the workspace

To evaluate sentiment analysis model: 'python3 -m src.sentimentAnalysis.evaluate'
to evaluate news classification model: 'python3 -m src.newsClassification.evaluate'

The data and models directories are not included in this repo, as they are too big in size. We will use the models trained with these scripts to experiment on different adversarial attacks
