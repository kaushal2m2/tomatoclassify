# tomatoclassify

The core of this project was using tensorflow to train a model using convolutional neural networks to be able to differentiate and predict whether tomato leaves had a certain disease.

## results

There were **10 classes** used in the [dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) found off kaggle. The accuracy currently ranges around **93%** in predicting the right class out of the given 10.

## What I'm currently doing: 

I'm looking into changes such as other optimizers like SGD, and potentially changing the type of model from Sequential in order to make the model more accurate.

### Note

In training, there is also train.py which uses a dataset of potatoes, which has 3 classes, and was trained in a similar model. That model achieved about 96% accuracy, but the only reason I did it was as a trial run for this project, so that model is not saved anywhere or used anywhere. It just exists for some valuable comments.

## Architecture

This uses a Sequential model, input augmentation, 7 convolutional, pooling, and batchnorm layers, and an adam optimizer in order to achieve these results.

I used Exponential Decay learning rate scheduling and a batch size of 64, which was able to boost the model from achieving 70% validation accuracy to 93% validation accuracy.

The model is saved in model/{model_version}. This model is served using flask to localhost port 5000, which makes the model available to the frontend.