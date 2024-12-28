import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import tqdm
import re

#tokenize all speeches by batches of up to 500 words containing full sentences.
#embeddings contains an embedding of 768 for each batch.
#text_batches contains all the batches in words and the matching text index.
def tokenize(text_column, tokenizer, batch_size):
    speech_list = text_column.astype(str).tolist()

    # Choosing Sentence Embedding Model
    model_name = tokenizer
    model = SentenceTransformer(model_name)

    # Define the maximum word limit for each batch
    max_words_per_batch = batch_size  # Set your desired word limit

    embeddings = []  # Save the embeddings
    batch_texts = []  # Save each batch text
    batch_article_mapping = []  # Save article number of each batch

    model.progress_bar = True

    # Separate counter variable
    article_counter = 0

    for text in tqdm.tqdm(speech_list):
        # Split the text into sentences (you may need to adjust this based on your data)
        sentences = text.split('.')  # Split on periods for example

        # Initialize variables for the current batch
        current_batch = []
        current_word_count = 0

        for sentence in sentences:
            # Calculate the word count of the sentence.
            words = sentence.split()
            sentence_word_count = len(words)

            # Check if adding the sentence would exceed the word limit
            if current_word_count + sentence_word_count <= max_words_per_batch:
                current_batch.append(sentence+'.')
                current_word_count += sentence_word_count
            else:
                # Append batch text
                batch_texts.append(" ".join(current_batch))
                # Process the current batch
                batch_embeddings = model.encode(" ".join(current_batch))
                embeddings.append(batch_embeddings)

                # Record the mapping between the batch and the source article
                batch_article_mapping.append(article_counter)

                # Reset the batch and word count for the next batch
                current_batch = [sentence+'.']
                current_word_count = sentence_word_count

        # Process the remaining sentences in the last batch
        if current_batch:
            batch_texts.append(" ".join(current_batch))
            batch_embeddings = model.encode(" ".join(current_batch))
            embeddings.append(batch_embeddings)
            batch_article_mapping.append(article_counter)

        article_counter += 1

    text_batches = pd.DataFrame({'article_index': batch_article_mapping, 'batch_texts': batch_texts})

    # Now, you have embeddings for all the batches and batch-to-article mapping
    return embeddings, text_batches



#embbed by sentences
def tokenize_sentences(text_column, tokenizer):
    speech_list = text_column.astype(str).tolist()

    # Choosing Sentence Embedding Model
    model_name = tokenizer
    model = SentenceTransformer(model_name)

    sentence_embeddings = []  # Save the embeddings
    sentence_number = []  # Save article number of each batch
    article_number = []
    sentence_texts = []
    model.progress_bar = True

    # Separate counter variable
    article_counter = 0

    for text in tqdm.tqdm(speech_list):
        # Split the text into sentences (you may need to adjust this based on your data)
        sentences = re.split(r'(?<=[.!?;])\s+(?![A-Z])', text) # Split on periods for example

        #count sentence number
        sentence_counter = 0

        for sentence in sentences:
            if len(sentence.strip()) >= 10:
                
                #save sentence 1.text 2.embedding 3.number.
                sentence_texts.append(sentence.strip())
                sentence_embeddings.append(model.encode(sentence.strip()))
                sentence_number.append(sentence_counter)
                # Increment the sentence counter for this article
                sentence_counter += 1

        # Save the article number for each sentence in this article
        article_number.extend([article_counter] * sentence_counter)
        # Increment the article counter for the next article
        article_counter += 1

    sentences_text = pd.DataFrame({'article_index': article_number, 'sentence_index': sentence_number, 'sentences_texts': sentence_texts})

    # Now, you have embeddings for all the batches and batch-to-article mapping
    return sentence_embeddings, sentences_text



#stack and save embeddings to a feather and csv file, and return rmbeddings stacked df.
def save_embeddings_list(embeddings_list, file_name, texts=None):
    stacked_embeddings=np.stack(embeddings_list) #stack
    df_embeddings = pd.DataFrame(stacked_embeddings) 
    df_embeddings.columns = df_embeddings.columns.astype(str) #needed for feather file

    df_embeddings.to_feather(file_name + "_embedds" + ".feather") #save embeedings to feater
    if texts is not None:
        texts.to_csv(file_name + "_text" + ".csv") #save texts to csv
    return df_embeddings