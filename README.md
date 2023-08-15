# Sentiment Analysis of Reviews
In this project, I chose to analyse the Sentiment140 dataset found [here](https://www.kaggle.com/datasets/kazanova/sentiment140). This dataset is composed of 1.6 million tweets each categorized as either negative or positive.
## Workflow
Once I had downloaded the data, I took the following steps:
1. Load data into a jupyter notebook and perform some basic data analysis (look for missing values, distributions, process data)
2. Prototype some models first in a jupyter notebook using Keras, saving run data with MLFlow
3. Build out a python package and convert code to PyTorch to make training more repeatable
4. Setup simple API endpoint and Docker container
5. Deployed model can now be accessed on an endpoint from within the container

## Why I Chose This Dataset
I wanted to showcase some NLP skills because text data is notoriously hard to wrangle as compared to pure numerical or categorical datasets. The dataset was also quite large so I knew I would need to leverage some sort of lazy loading setup because I was going to be performing all training on my machine. This gave me a chance to build out a python pacakge for training on larger and larger subsets of the data if time allows in the future. With a simpler dataset, I wouldn't have really had a reason to do a lot of these extra steps as the workflow is more straighforward.
## Challenges
My biggest challenge was to deal with the long training times on this kind of data. I originally trained on a smaller subset of the data (100k samples), but after playing around with hyperparameters for a bit too long, I realised that the only way to increase performance was to increase the size of the input data. 
## Notes
I experimented with preprocessing the data, but the embeddings produced balooned the datasize. I opted instead to do this in a custom dataset to avoid taking up too much space on disk. This also makes it easier in the future if I decided to add an embedding layer to the model to train a custom embedding along with the rest of the model params.