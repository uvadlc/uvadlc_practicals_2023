# Assignment 3, Part 3: GPT2

This folder contains the template code for implementing your own GPT2 model. This corresponds to Question .. to .. in the assignment. We will train the model on predicting next characters. The code is structured in the following way:

* `dataset.py`: Contains a function for creating and preparing the dataset.
* `gpt.py`: Contains template classes for the Encoder, Decoder, Discriminator and the overall Adversarial Autoencoder.
* `train.py`: Contains the overall procedure of assignment, it parses terminal commands by user, then it sets the hyper-parameters, load dataset, initialize the model, set the optimizers and then
  it trains the adversarial auto-encoder and saves the network generations for each epoch.   
* `unittests.py`: Contains unittests for the Encoder, Decoder, Discriminator networks. It will hopefully help you debugging your code. Your final code should pass these unittests.

A lot of code is already provided to you. Try to familiarize yourself with the code structure before starting your implementation. 
Your task is to fill in the missing code pieces (indicated by `NotImplementedError` or warnings printed). The main missing pieces are:

* In `gpt.py`, you need to implement a part of the `forward` function for the `CausalSelfAttention` block. You need to implement the part which calculates the attention weights. The rest of the architecture is given and should not be necessary to change. Furthermore, you need to implement the generate function of the GPT model. This function is used to forward some initial text through the model, sample from the output distribution and converts the indices back to text.
  
Default hyper-parameters are provided in the `ArgumentParser` object of the respective training functions. Feel free to play around with those to familiarize yourself with the effect of different hyper-parameters. Nevertheless, your model should be able to generate decent images with the default hyper-parameters.
The training time for descent performance, with the default hyper-parameters and architecture, is less than 30 minutes on a NVIDIA GTX1080Ti (GPU provided on Lisa).

The `generate.py` file can be used to load pretrained gpt2 weights and generate sentences based on some context. Try to play with the parameters.