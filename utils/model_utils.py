from sklearn.metrics import classification_report
from tabulate import tabulate
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from tqdm import tqdm
from utils.tokenize_utils import save_embeddings_list
from sentence_transformers import SentenceTransformer


#create deviations column
def deviations_col(df, score_col, prediction_col):
    deviations=pd.DataFrame(abs(df[score_col]-df[prediction_col]))
    df = df.assign(deviations=deviations)
    return df

#tag as 1 when larger then number. else 0.
def binary_tag(df, threshold, new_col_name):
    df.loc[:, new_col_name] = pd.DataFrame(np.where(df['deviations'] > threshold, 1, 0))
    return df

#return only tagged most deviated sorted reviews where prediction>overallscore
def sorted_tagged_and_larger_pred(df, prediction_col, score_col, deviations_col):
    sorted = df[df[prediction_col]>df[score_col]].sort_values(by=deviations_col, ascending=False)
    return sorted

#function for trees based regression models. printing train & test mse.
def score_summary(y_train, y_train_pred, y_test, y_test_pred, scorer):
    score_train = scorer(y_train, y_train_pred)
    score_test = scorer(y_test, y_test_pred)
    print(f"{scorer.__name__} (Train):", score_train)
    print(f"{scorer.__name__} (Test):", score_test)
    return score_train, score_test


def summary_df(data, prediction_col, score_col, tag_threshold, tagged_col):
    new_df= data[['Title','Review', 'OverallScore']]
    new_df['prediction'] = prediction_col
    new_df= deviations_col(new_df, score_col, 'prediction')
    new_df= binary_tag(new_df, tag_threshold, tagged_col)
    new_df= sorted_tagged_and_larger_pred(new_df, 'prediction', score_col, 'deviations')
    return new_df

def tagged_df(data, tag_col):
    new_data=data[data[tag_col]==1]
    return new_data

def show_metrics(y_test, y_pred):
    report = classification_report(y_test, y_pred)
    report_lines = report.split('\n')
    # Remove empty lines and header/footer lines
    report_lines = [line for line in report_lines if line.strip().startswith('precision')==False]
    # Convert the classification report lines into a list of lists
    table_data = [line.split() for line in report_lines]
    # Add headers for the table
    headers = ["Class", "Precision", "Recall", "F1-Score", "Support"]
    table_data.insert(0, headers)
    # Format the table using tabulate
    table = tabulate(table_data, headers="firstrow", tablefmt="pretty")
    print(table)


class EvidenceExtractor:
    def __init__(self, sentence_breaker, anchor_sentences, model_name):
        self.sentence_breaker = sentence_breaker #This is a function used to break long text into sentences
        self.anchor_sentences = anchor_sentences 
        self.model = SentenceTransformer(model_name)
        self.sentence_embeddings=[]

        self.anchor_sentences_embeddings = self.model.encode(self.anchor_sentences)
        self.anchor_sentences_embeddings = self.anchor_sentences_embeddings / np.linalg.norm(self.anchor_sentences_embeddings, axis=1, keepdims=True)

    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        out = []
        for speech in tqdm(X, desc="Processing Text", unit="speech"):
            sentences = self.sentence_breaker(speech)
            sentence_embeddings = self.model.encode(sentences)
            self.sentence_embeddings.append(sentence_embeddings)
            sim_matrix = cosine_similarity(self.anchor_sentences_embeddings, sentence_embeddings)
            features_vector = sim_matrix.max(axis=1)
            out.append(features_vector)
        #all_embeddings = np.vstack(self.sentence_embeddings)
        #save_embeddings_list(all_embeddings, 'speech_sen') #save sentence embedds
        return pd.DataFrame(np.stack(out), index=X.index)

# not splitting on first capital letter in sentence
def sentence_breaker(text):
    sentences= re.split(r'(?<=[.!?;])\s+(?![A-Z])', text)
    return sentences

#allowing split on first capital letter- better.
def sentence_breaker2(text):
  sentences = re.split(r'[.!?;]\s+', text)
  return sentences

#this one also completes the missing mark.
def sentence_breaker3(text):
    sentences = re.split(r'(?<=[.!?;])\s+', text)
    return sentences


#save cumulative texts
def cumulate_sentences(data, text_num, sentence_breaker=sentence_breaker3):
# Initialize an empty list to store cumulative texts
    list_cumulative_sen = []

    # Create cumulative texts by appending sentences iteratively
    cumulative_text = ""
    for sentence in sentence_breaker(data[text_num]):
        cumulative_text += sentence + " "  # Concatenate sentences with space
        list_cumulative_sen.append(cumulative_text.strip())
    out = pd.Series(list_cumulative_sen)
    return out
