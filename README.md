# Balanced-DEC

Keras and CUDA implementation for GRSL paper:

* Unsupervised Land-Cover Segmentation Using Accelerated Balanced Deep Embedded Clustering.

## Usage
1. Install [Keras>=2.0.9], scikit-learn  
```
pip install keras scikit-learn   
```
2. Make sure you have CUDA for the CUDA part
```
3. Clone the code to local.   
```
git clone https://github.com/AhmadObeid/Balanced-DEC.git

```

4. Download the datasets.    



4. Run experiment on MNIST.   
`python DEC.py --dataset mnist`   
or (if there's pretrained autoencoder weights)  
The DEC model will be saved to "results/DEC_model_final.h5".

5. Other usages.   

Use `python DEC.py -h` for help.

## Results

```
python run_exp.py
