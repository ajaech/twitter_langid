# twitter_langid
For more information please see our paper 
[Hierarchical Character-Word Models for Language Identification](https://arxiv.org/abs/1608.03030).


### Getting Started

To train a model run the command

`python langid.py path/to/outdir`

For some simple visualization of a trained model run

`python langid.py --mode=debug path/to/outdir`

To evaluate a trained model run 

`python langid.py --mode=eval path/to/outdir`

### Data

The data directory holds an example input file created from Wikipedia sentence fragments. The file is saved in tab separated format. We partition the data according to the last digit of the id number in the data file. Separate lines are used for training, validation, and testing.
