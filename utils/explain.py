import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import shap

#mapper for different classes from str to integers
def map_decision(decision):
    decision_mapping = {"decrease": 0, "increase": 1, "no_change": 2}

    if decision not in decision_mapping:
        
        raise ValueError(f"Invalid decision: {decision}")
    return decision_mapping[decision]


#gets the prediction of a speech and returns the most important features in this decision (anchors)
def specific_anchors_explainer(X_explain, model, anchors, explainer=shap.TreeExplainer, mapper=map_decision, top_num=5):

    '''
    X_explain: the whole feature values row of one speech
    model: the fitted model
    '''

    X_explain=np.array(X_explain).reshape(1, -1)

    #predict
    y_pred = model.predict(X_explain)

    #save decision num
    explain_class= mapper(y_pred[0])

    # Get SHAP values for the prediction
    explain = explainer(model)
    shap_values = explain(X_explain)

    #choose the class to return best anchors for
    target_class_shap_values=  shap_values[:,:,explain_class]

    # Sort SHAP values
    sorted_shap_values = np.abs(target_class_shap_values.values).argsort()[::-1]

    # Select the top 5 features
    top_n_features = sorted_shap_values[0,:top_num]
    top_n_features= pd.Series(anchors).iloc[top_n_features.tolist()]

    return top_n_features.tolist()

#input: sentence breaker, most important anchor sentences and model name.
#output: df of the most important sentences in the text (by most important features).
class HighlightsExtractor:
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
        sentences = self.sentence_breaker(X)
        sentence_embeddings = self.model.encode(sentences)
        self.sentence_embeddings.append(sentence_embeddings)
        sim_matrix = cosine_similarity(self.anchor_sentences_embeddings, sentence_embeddings)
        max_sim_per_feature = np.argmax(sim_matrix, axis=1)  # Get indices of max similarity
        sentences= pd.Series(sentences)[max_sim_per_feature]
        return pd.DataFrame(sentences)

