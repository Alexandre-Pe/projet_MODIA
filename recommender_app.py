import argparse
import gradio as gr
import joblib
import pickle
import lzma
import tarfile
import os

# after training add : 
# import dill
# with open('vectorizer.joblib','wb') as io:
#     dill.dump(tfidf ,io)

# from joblib import dump, load
# #pickle model to disk
# dump(rf, 'rf_model.joblib', protocol=2) 
# #loading saved model
# estimator = load('rf_model.joblib')

from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')
# Download stopwords list

stop_words = set(stopwords.words('english'))  | {
    "recipe",  "flour",
    "made", "food", "cup", "give", 'make', 'chicken',
    'used', 'mustard', 'pesto', 'this', 'that', 'cake'
}

# Interface lemma tokenizer from nltk with sklearn
class StemTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`', '&#039;']
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
    def __call__(self, doc):
        return [self.stemmer.stem(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]




def predict_sentiment(com):
    pred = model.predict([com])
    pred_proba = model.predict_proba([com])
    if pred == "negative" : return pred + "(predicted proba = "+ str(pred_proba[0][0]) + ")"
    else : return pred + "(predicted proba = "+ str(pred_proba[0][1]) + ")"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str,
                        default="rf_model.joblib", help="weights path")
    args = parser.parse_args()

    tokenizer=StemTokenizer()
    token_stop = tokenizer(' '.join(stop_words))
    tfidf = TfidfVectorizer(stop_words=token_stop, tokenizer=tokenizer, max_features=2000)

    if os.path.isfile("data/pipeline_part2.pkl")==False : 
        with lzma.open("pipeline_part2.tar.xz") as fd:
            with tarfile.open(fileobj=fd) as tar:
                content = tar.extractall('data/')

    with open('data/pipeline_part2.pkl', 'rb') as fo:  
        model = joblib.load(fo)

    comment_input = gr.inputs.Textbox(lines=5, label="Enter a comment")

    # Create the output component (text output)
    sentiment_output = gr.outputs.Textbox(label="Sentiment")

    # Create the Gradio interface
    gr.Interface(fn=predict_sentiment, 
                 inputs=comment_input, 
                 outputs=sentiment_output,
                 ).launch(debug=True, share=True)
