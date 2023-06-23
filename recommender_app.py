import argparse
import gradio as gr
import joblib
import pickle
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import dill

# after training add : 
# import dill
# with open('vectorizer.joblib','wb') as io:
#     dill.dump(tfidf ,io)

# from joblib import dump, load
# #pickle model to disk
# dump(rf, 'rf_model.joblib', protocol=2) 
# #loading saved model
# estimator = load('rf_model.joblib')


def predict_sentiment(com):
    com = tfidf.transform([com])
    pred = model.predict(com)
    pred_proba = model.predict_proba(com)
    if pred[0] == 0 : return "negative (predicted proba = " + str(pred_proba[0][0]) + ")"
    else : return "positive (predicted proba = " + str(pred_proba[0][1]) + ")"

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights_path', type=str,
                        default="data/rf_model.joblib", help="weights path")
    parser.add_argument('--vectorizer_path', type=str,
                        default="data/vectorizer.joblib", help="vectorizer path")
    args = parser.parse_args()

    model = joblib.load(args.weights_path)
    with open(args.vectorizer_path,'rb') as io:
        tfidf=dill.load(io)

    comment_input = gr.inputs.Textbox(lines=5, label="Enter a comment")

    # Create the output component (text output)
    sentiment_output = gr.outputs.Textbox(label="Sentiment")

    # Create the Gradio interface
    gr.Interface(fn=predict_sentiment, 
                 inputs=comment_input, 
                 outputs=sentiment_output,
                 ).launch(debug=True, share=True)
