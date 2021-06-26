# Hybrid Training of Neaural Network
Artificial life projects for the the University of Sussex AI and Adaptive systems Master 2020.

The project focus on hybrid training of feedforward neural network, a combination of genetic algorithm and classic multilayer perceptrons.

# Abstract

This paper shows how the speed of convergence of an artificial neural network can be improved using a genetic algorithm. Standard gradient-descent algorithms, such as backpropagation, suffer from various problems connected to the random initialization of the network’s weights. Using the global-search ability of a genetic algorithm, it is possible to find an optimal starting point. The starting set of weights is then fine-tuned by the gradient-descent algorithm. The paper shows how a simple genetic algorithm, a microbial genetic algorithm, can effectively reduce the number of epochs needed to learn a task. The experiment has been conducted on three different benchmark tasks: a XOR problem, an NMN encoder problem and a two-spirals problem. The real benefits of this hybrid approach become apparent when the error landscape presents many local minima, which present a problem for gradient descent. The use of this hybrid approach has been proved to be more efficient than standard methods in different fields, such as finance and medicine.

Materials used during research:

- possible tasks to test a simple NN
- material used for my paper on hybrid training 
