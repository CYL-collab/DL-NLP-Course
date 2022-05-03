import pickle
from sklearn.feature_extraction.text import CountVectorizer

if __name__ == "__main__":
    with open("LDA/doc.pkl", "rb") as f:
        doc = pickle.load(f)
    print(doc)