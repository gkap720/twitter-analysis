# Sentiment Analysis of Reviews
In this project, I chose to analyse the Sentiment140 dataset found [here](https://www.kaggle.com/datasets/kazanova/sentiment140). This dataset is composed of 1.6 million tweets each categorized as either negative or positive.
## Workflow
Once I had downloaded the data, I took the following steps:
1. Load data into a Jupyter notebook and perform some basic data analysis (look for missing values, distributions, process data)
2. Prototype some models first in a jupyter notebook using Keras, saving run data with MLFlow
3. Iterate on the model by looking at learning curves and diagnosing whether the model is over-/underfitting
4. Build out a python package and convert code to PyTorch to make training more repeatable
5. Setup simple API endpoint and Docker container
6. Deployed model on the container

## Why I Chose This Dataset
I wanted to showcase some NLP skills because text data is notoriously hard to wrangle as compared to pure numerical or categorical datasets. The dataset was also quite large so I knew I would need to leverage some sort of lazy loading setup because I was going to be performing all training on my machine. This gave me a chance to build out a python package for training on larger and larger subsets of the data if time allows in the future. With a simpler dataset, I wouldn't have really had a reason to do a lot of these extra steps as the workflow is more straighforward.
## Model Choice
For this type of data, I chose a relatively simple LSTM model. LSTMs are well-suited to sequence data and help to mitigate the vanishing gradient problem. I used some fully connected Dense layers after the two LSTM layers and a final predictive layer which outputs a probability of the tweet being positive or negative. I used tanh activations throughout as these are known to help mitigate the vanishing gradient issue as well. Another option for this kind of data would be to conduct transfer learning on top of an already-trained model (a Transformer model for example) which would likely produce better results, but of course require far less work to implement.
## Challenges
My biggest challenge was to deal with the long training times on this kind of data. I originally trained on a smaller subset of the data (100k samples), but after playing around with hyperparameters for a bit too long, I realised that the only way to increase performance was to increase the size of the input data. After doubling the training set, I gained 6% accuracy right away, but of course the training times increased as well. This bottleneck limited my ability to iterate on the model and try out other hyperparameters and setups. If given more time, I would have done all the training on the cloud with a GPU-accelerated machine which would have mitigated most of these issues. This would also be my preference in a production environment.
## Future Development and Maintenance
If this project were to be carried on into the future, I would set up a remote MLFlow server to host runs of the model so that different members of the team could access the findings and also so that we could easily keep a versioning system for our model runs (Weights and Biases would serve a similar purpose). As the model is used in production, I would regularly evaluate it against new data coming into the platform (in this case new tweets). If the performance starts to drift, the model could then be retrained with newer data. Right now, the Docker container is somewhat "hardcoded" so I would also take steps to allow the container to be more configurable. This would just involve grabbing model weights from a remote source (an S3 bucket for example or the remote MLFlow instance) instead of directly from within the container. The path to the remote weights could then be passed as an environment variable to the container when it's run instead of requiring changes to the Dockerfile or needing a rebuild.
## Notes
I experimented with preprocessing the data, but the embeddings produced balooned the datasize. I opted instead to perform the embeddings in a custom dataset to avoid taking up too much space on disk. This also makes it easier in the future if I decided to add an embedding layer to the model to train a custom embedding along with the rest of the model params.
## Usage
Before running anything:
1. Create a `data` directory in the root of the project and download the dataset into this folder
2. Install the package with `pip install .` while in the root of the project

All of the following commands should be executed in the root of the project.
### Create train, test, val sets
`python -m src.data.process`

By default this will create a train set of 50k samples. You can set a larger size by specifying a `-size` flag like so:

`python -m src.data.process -size 100000`
### Run model training
`python -m src.models.train`

Data about the training will be stored in the `mlruns` directory.
### View visualizations of the training run
`mlflow ui`

If there is not enough time to train the model, check out some example runs by running:

`mlflow ui --backend-store-uri example_runs`
### Evaluate model on test set
`python -m src.models.eval`
### Create Docker image
`docker build . --tag sentiment-analysis`

(This can take awhile)
### Run Docker image
`docker run -d -p 80:80 sentiment-analysis`
### Access endpoint
Paste this into the browser: `localhost:80?input_str=python is great`
