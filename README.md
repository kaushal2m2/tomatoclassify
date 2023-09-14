# tomatoclassify

The core of this project was using tensorflow to train a model using convolutional neural networks to be able to differentiate and predict whether tomato leaves had a certain disease.

## results

There were **10 classes** used in the [dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village) found off kaggle. The accuracy currently ranges around **70%** in predicting the right class out of the given 10.

This uses a Sequential model, input augmentation, 7 convolutional and pooling layers, and an adam optimizer in order to achieve these results.

## What I'm currently doing: 

I'm looking into changes such as other optimizers like SGD, adding learning rate scheduling, and potentially changing the type of model from Sequential in order to make the model more accurate.

### Note

In training, there is also train.py which uses a dataset of potatoes, which has 3 classes, and was trained in a similar model. That model achieved about 96% accuracy, but the only reason I did it was as a trial run for this project, so that model is not saved anywhere or used anywhere. It just exists for some valuable comments.