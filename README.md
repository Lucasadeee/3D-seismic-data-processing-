
Technological advances in oil and gas reservoir characterization such as 3D seismics attributes enriched the subsurface’s description made by specialists. Nevertheless, the analysis of this now huge volume of data became a complex task. This work explores the use of convolutional neural networks for seismic facies classification, one of the steps of reservoir characterization. Through a sampling method that captures spacial informationof seismic data, the models produced were applied in both synthetic data of the Stanford VI-E reservoir and in a benchmark based on the F3 block, which is part of a real reservoir. Compared to other models in the same benchmark, the classifiers produced here had similar results, with over 90% class accuracy on some instances. The sampling method is also flexible to use in practical cases.

## Setup
After you've created your python virtual environment, run:
```
pip install -r requirements.txt
```
## install setup file
run python setup.py install from cmd

### Stanford VI-E
The raw Stanford VI-E dataset used in this repo is included. To generate the training examples, use the following script (see --help for arguments).
```
python src/data/make_dataset.py
```

### F3-Block
The F3 data used in this repo is avaiable [here](https://github.com/olivesgatech/facies_classification_benchmark). 
numpy arrays for seismic data and their labels can be downloaded from the link above.
To generate the training examples, use the following script (see --help for arguments).

```
python src/data/make_dataset_f3.py
```

## Running
Jupyter notebooks are located in ```/notebooks```. The trained models are stored in ```/models```.