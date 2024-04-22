# AdversarialAttacks

## Download Data and Models:
Download data:  
https://drive.google.com/file/d/1wAx51b1fm0N4fB3tlVeSc28mE_scBOYM/view?usp=sharing  

Download models:  
https://drive.google.com/file/d/1bsNfdSWD3VIKe1pE7qhB9y1XWwkl53m3/view?usp=sharing  

Download the data and models and put them in the root of the project. They should be kept as they are (inside a "*data*" and "*models*" directory each).

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
To evaluate the **models**, the following command should be used:
```
python -m src.<model>.<task>.evaluate
```
Therefore, the command to evaluate the different **models** for the different tasks is as follows:
```
python -m src.MLP.sentimenAnalysis.evaluate
python -m src.MLP.newsClassification.evaluate
python -m src.LSTM.sentimenAnalysis.evaluate
python -m src.LSTM.newsClassification.evaluate
```
  
To evaluate the **random synonym attacks**:
```
python -m src.synonymAttackRandom.sentimentAnalysis.evaluate
python -m src.synonymAttackRandom.newsClassification.evaluate
```

## Note
There are different models. By default, the train and evaluate have the model with the best performance. However, if you want to train and / or evaluate a specific model, you can change the model path in the train and evaluate files.