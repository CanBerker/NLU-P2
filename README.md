# NLU-P2
Natural Language Understanding - Project 2

## Requirements
1. Wikipedia word embeddings 
    1.1. Download the word embeddings [http://nlp.stanford.edu/data/glove.6B.zip]
    1.2. Unzip word embeddings file
2. Check parameters
    2.1.
## How to run in Leonhard
1. Activate python module
    	module load python_gpu/3.6.0
   Verify it is loaded:
	module list
2. Create the folder where models will be saved
	mkdir checkpoint
3. Launch using
	bsub -n 1 -R rusage[mem=5000,ngpus_excl_p=1] python src/main.py -s checkpoint/ -gp /cluster/home/marenato/glove/wikipedia/glove.6B.50d.txt 
