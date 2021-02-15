# Balanced-DEC

## Usage
There are 2 modes of experimentation:  
1. Load the fully trained network, and use it in a prediction-only fashion
2. Load only the weights of the autoencoder, and go through the DEC algorithm

To run the code:

```
python main.py --file <file> --mode <mode> typ <type> 
```
The available files are:
```
* S, SA, PC, PU, pavia_small
```
The available modes are:
```
* BDEC, DEC
```
The available types are:
```
* predict, train
```

## Results
Results will be printed in the command window, and can be seen in the results directory

## Using your fetched representatives from /search

The indices directory contains suggested indices that were used by the authors, and should give you the same results of the paper.
To use your own indices (i.e. after finding them using the search step), copy and past them in the indices files and save.

