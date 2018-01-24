### A neural net in Swift.

I'm trying to understand how Neural Networks function using resources like [this](http://neuralnetworksanddeeplearning.com/chap1.html) and [this](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi). Once I started thinking I understood what was going on, I wanted to see if I could build a neural network in Swift.

Most of the code defines Matrix and Vector classes that can be created and operated on, mostly so I could multiply, add, and equate vectors and matrices.

Some of the code, however, sets up layers, weights, and biases in a Network class.

The Network.feedForward method runs a single iteration of the Network object given a set of inputs. Right now, weights and biases are initialized to a random double between 0.0 and 1.0. Eventually I want to implement backpropagation and learning.

If you stumble upon this code and want to contribute, please feel free to send me either critiques of my understanding of neural networks or critiques of the "swiftiness" of my code. Please don't send me pull requests.
