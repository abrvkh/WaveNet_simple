# WaveNet_simple
A simple implementation of the WaveNet model for time series forecasting

This is a version of the WaveNet model as used for time series forecasting, based on the WaveNet model by DeepMind (https://arxiv.org/pdf/1609.03499.pdf). 

The model is similar to the one proposed in https://arxiv.org/pdf/1703.04691.pdf, but with several modifications.

- The idea is to use every time series we want to use in the forecasting, as a separate channel in the input, so that we use nCond channels, where nCond is the total number of inputs. In this way hopefully the time series will get combined in a sufficiently non-linear manner allowing to learn dependencies in between them. 

- Each layer then computes nFilters (number of filters we use) convolution with each of the channels, and subsequently sums them and passes them through the non-linearity. 

- After each layer we thus end up with a time series with nFilters channels. 

- If we then use the nFilters=nCond, and compute the error between the output and the input, we would get a forecast for each of the time series simultaneously (and thus being hopefully quite efficient). 

- The time series used now is the Lorenz curve and a one-day-ahead sampling.

Disclaimer: This is a first trial version of this kind of implementation. The implementation needs to be carefully checked to make sure it avoids the look-ahead bias; furthermore, using the number of channels in this way could make it harder for the model to differentiate between which inputs are needed to improve the forecast and which are irrelevant, resulting in a larger error. 
