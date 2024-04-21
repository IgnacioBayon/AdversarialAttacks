# AdversarialAttacks

## Download Data and Models:
Download the data and the trained models from the following link:  
https://drive.google.com/file/d/1Vjo-OBFpRGPaKk9KtlTgoIKEPWvuW18i/view?usp=drive_link

## How to train:
The code is organized in folders. Therefore, the structure of the command to train the model is as follows:
```
python -m src.<model>.<task>.train
```
Therefore, the command to train the different models for the different tasks is as follows:
```
python -m src.MLP.sentimenAnalysis.train
python -m src.MLP.newsClassification.train
python -m src.LSTM.sentimenAnalysis.train
python -m src.LSTM.newsClassification.train
```


## How to evaluate
To evaluate the models, the following command should be used:
```
python -m src.<model>.<task>.evaluate
```
Therefore, the command to evaluate the different models for the different tasks is as follows:
```
python -m src.MLP.sentimenAnalysis.evaluate
python -m src.MLP.newsClassification.evaluate
python -m src.LSTM.sentimenAnalysis.evaluate
python -m src.LSTM.newsClassification.evaluate
```

## Note
There are different models. By default, the train and evaluate have the model with the best performance. However, if you want to train and / or evaluate a specific model, you can change the model path in the train and evaluate files.