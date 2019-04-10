# RNN-with-LSTM
## RNN with LSTM implementation for bit-wise addition
## 1 Algorithm and Python Code
Here we implement a simple RNN with LSTM to carry wise bitwise addition. The concept of storing the
carry bit from the previous addition is learned through the RNN. The code was built with reference to the
algorithm mentioned here. However, the said algorithm does not consider the final carry generated after
the addition of all the bits in the string, i.e. the final result is the same length as the summands even when
it should have 1 more bit as the MSB. The same has been taken care of in this code as we can verify at the
end of the code.
### 1.1 Dataset Initialization
We generate our datasets using a simple function that takes the bit string length, batch size and number
of summands as inputs. The functions generate 2 random numbers within the given range of bit lengths
and add them up. The summands and the results are then converted to binary strings and the process is
repeated until the entire batch is created. The final dataset is returned for training.
### 1.2 Building the model
We use a basic LSTM cell from tensorflow. The cell function function that given input and previous state
returns tuple (output, state) where state is the state passed to the next timestep and output is the tensor
used for infering the output at timestep. For example for LSTM, output is just hidden, but state is memory
+ hidden. We then create an initial state, initializing it with zeros. Using dynamic-rnn, we create an RNN
model providing it with the cell and inputs. We finally project the output from this RNN into a layer same
as the output size.
### 1.3 Optimizer and Loss
We compute the error through elementwise cross entropy. Then we use and Adam optimizer for optimiza-
tion. Finally we use the standard formula to calculate the accuracy.
### 1.4 Operation
We run sessions with separate datasets for training and validation. Then after the total number of epochs,
we use the model on another test data. This takes care of overfitting. The predictions are based on the
Bayes’ classifier(1 in case of >0.5).
## 2 Training and Testing Performance
The training accuracy increases to 100% in 10 iterations. The test accuracy is equal to 100%.
Epoch 0, train error: 0.63, valid accuracy: 65.8 %
Epoch 1, train error: 0.58, valid accuracy: 72.9 %
Epoch 2, train error: 0.42, valid accuracy: 86.5 %
Epoch 3, train error: 0.19, valid accuracy: 93.6 %
Epoch 4, train error: 0.10, valid accuracy: 100.0 %
Epoch 5, train error: 0.04, valid accuracy: 100.0 %
Epoch 6, train error: 0.02, valid accuracy: 100.0 %
Epoch 7, train error: 0.01, valid accuracy: 100.0 %
Epoch 8, train error: 0.01, valid accuracy: 100.0 %
Epoch 9, train error: 0.01, valid accuracy: 100.0 %
Test accuracy: 100.0 %
We can use the model to verify our own input as follows:-
Enter 1st string: 1 0 1
Enter 2nd string: 1 1 1
Predicted output:
[1. 1. 0. 0.]

## 3 References
• https://appliedmachinelearning.blog/

• http://www.wildml.com/2015/09/recurrent-neural-networks-tutorial-part-2-implementing-a-language-
model-rnn-with-python-numpy-and-theano/

• https://www.coursera.org/lecture/nlp-sequence-models/recurrent-neural-network-model-ftkzt
