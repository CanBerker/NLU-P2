# NLU-P2
Natural Language Understanding - Project 2
Source structure:
/data --> training, validation, and test datasets
/src
    /strategies --> contains all the code to train the models
    /extractors --> contains all the code to utilize the trained models

## Requirements
1. Wikipedia word embeddings 
    1.1. Download the word embeddings [http://nlp.stanford.edu/data/glove.6B.zip]
    1.2. Unzip word embeddings file
2. Set up project which has been developed using python3 and assumes a working installation of Tensorflow and Keras.
    2.0. Install python dependencies
        If on UNIX check that h5py is install for keras
	```apt-get install libhdf5-dev  python-h5py```
        Then install requirements:
	```pip3.6 install -r requirements.txt```
    2.1. Execute the setup script:
        ``` python3.6 setup.py ```
    2.2. Run
        ``` python3.6 src/main.py --help```
         This should print the following parameters:
        ``` 
        usage: main.py [-h] [-t TPATH] [-v VPATH] [-s SPATH] [-m MODEL] [-g]
                       [-gp GLOVE_PATH] [-en]
                       [-ea [ENSEMBLE_APPROACHES [ENSEMBLE_APPROACHES ...]]] [-vt]
                       [-eth] [-ct] [-mp MODEL_PATH] [-lc_mp LSTM_CLASS_MODEL_PATH]
                       [-lm_mp LANG_MODEL_MODEL_PATH]
        
        optional arguments:
          -h, --help            show this help message and exit
          -t TPATH, --training_dataset TPATH
                                Training dataset path
          -v VPATH, --validation_dataset VPATH
                                Validation dataset path
          -s SPATH, --save_model_path SPATH
                                Model directory
          -m MODEL, --model_to_train MODEL
                                Model to be trained
          -g, --use_gpu         Use GPU for training
          -gp GLOVE_PATH, --glove_path GLOVE_PATH
                                Glove file path
          -en, --use_ensemble   Use ensemble of methods
          -ea [ENSEMBLE_APPROACHES [ENSEMBLE_APPROACHES ...]], --ensemble_approaches [ENSEMBLE_APPROACHES [ENSEMBLE_APPROACHES ...]]
                                Ensemble approaches (none defaults to
                                SentimentTrajectory)
          -vt, --validate_on_testset
                                Perform validation on test set
          -eth, --validate_on_eth_set
                                Perform validation on ETH test set
          -ct, --continue_training
                                Continue training
          -mp MODEL_PATH, --model_cont_training_path MODEL_PATH
                                Model path to continue training
          -lc_mp LSTM_CLASS_MODEL_PATH, --lstm_class_model_path LSTM_CLASS_MODEL_PATH
                                LSTM classifier model path to be used for ensembling.
          -lm_mp LANG_MODEL_MODEL_PATH, --lang_model_path LANG_MODEL_MODEL_PATH
                                Language model model path to be used for ensembling.
        
                 usage: main.py [-h] [-t TPATH] [-v VPATH] [-s SPATH] [-m MODEL] [-g]
                                   [-gp GLOVE_PATH] [-en]
                                   [-ea [ENSEMBLE_APPROACHES [ENSEMBLE_APPROACHES ...]]] [-vt]
                                   [-ct] [-mp MODEL_PATH] [-lc_mp LSTM_CLASS_MODEL_PATH]
                                   [-lm_mp LANG_MODEL_MODEL_PATH]
            ```
3. For running the Ensemble model execute the following command:
    ```
    python3.6 src/main.py -s checkpoint/ -gp ../glove/wikipedia/glove.6B.50d.txt -en -lc_mp trained_model/lstm_classifier_20e.hdf5 -lm_mp trained_model/lang_model_20e.hdf5 -eth
    ```


## How to run in Leonhard
1. Activate python module
    	module load python_gpu/3.6.0
   Verify it is loaded:
	module list
2. Create the folder where models will be saved
	mkdir checkpoint
3. Launch using
	bsub -n 1 -R rusage[mem=5000,ngpus_excl_p=1] python src/main.py -s checkpoint/ -gp /cluster/home/marenato/glove/wikipedia/glove.6B.50d.txt 
