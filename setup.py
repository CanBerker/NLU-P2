import os

def nltk_deps():
    import nltk
    nltk.download('punkt')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('vader_lexicon')

if __name__ == "__main__":
    nltk_deps()
