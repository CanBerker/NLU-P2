import nltk
import os

def pip_depts():
    pip_cmd = "pip install -r requirements"
    os.execute(pip_cmd)

def nltk_deps():
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')

if __name__ == "__main__":
    pip_deps()
    nltk_deps()