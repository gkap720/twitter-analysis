# Sentiment Analysis of Reviews

## Workflow
1. Load data into a jupyter notebook and perform some basic data analysis (look for missing values, distributions, process data)
2. Prototype some models first in the notebook with Keras, saving run data with MLFlow
3. Build out package to make training more repeatable
4. Setup simple API endpoint and docker container
5. Deploy

## Notes
I experimented with preprocessing the data, but the embeddings produced balooned the datasize. I opted instead to do this in a custom dataset to avoid taking up too much space on disk. This also makes it easier in the future if I decided to add an embedding layer to the model to train a custom embedding along with the rest of the model params.