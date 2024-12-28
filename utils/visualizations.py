#NEW IN USE:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.lines as mlines
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import WordCloud, STOPWORDS

#styled count plot of ratings column.
def styled_count_plot(data, column, title=None):
    data = data.dropna(subset=[column])
    # Convert to integers in the copied DataFrame
    data[column] = data[column].astype(int)
    # Create the value counts for 'OverallScore', sort by index in ascending order
    value_counts = data[column].value_counts().sort_index(ascending=True)
    # Set the figure size
    plt.figure(figsize=(12, 4))
    # Create the bar plot
    ax = value_counts.plot.bar(color='skyblue', edgecolor='black', width=0.7)
    #set title
    if title is not None:
        plt.title(title)
    else:
        plt.title(f'Count of {column}', fontsize=16)
    #more options:
    plt.xlabel(column)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Apply the comma formatting to the y-axis ticks directly
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))
    plt.tight_layout()
    plt.show()


#styled Ticked Histogram
def styled_hist(col, bins_n):
    plt.hist(col, bins=bins_n, edgecolor='black')

    # Add labels and title
    plt.xlabel('Word Length')
    plt.ylabel('Number of Speeches')
    plt.title('Number Of Words in FED Speeches')

    # Add grid lines
    plt.grid(True)

    # Format y-axis labels with commas for thousands
    plt.gca().yaxis.set_major_formatter(ticker.StrMethodFormatter('{x:,.0f}'))

    # Show the plot
    plt.tight_layout()
    plt.show()

#style data with review column
def read_speech(data, speech_num=None, word_lim=None):
    pd.set_option('max_colwidth', 1000)
    pd.set_option("display.colheader_justify", "left")
    if speech_num is not None:
        data=pd.Series(data.iloc[speech_num])
    
    if word_lim is not None:
        data = data.str.split().str[:word_lim].apply(lambda x: ' '.join(x))
          
    out_data = pd.DataFrame({'speech_text': data}).style.set_properties(**{'text-align': 'left'})

    return out_data

def common_words_count(column, stop_words=False, top_num=60):

    #join all words to one string
    all_titles = " ".join(column)
    #split each word in a list
    all_title_words= pd.Series(all_titles.lower().split())

    #with/without stop words:
    stop_words= ['and', 'the', 'of', 'in', 'for', 'on', 'to', 'a', 'an', 'at', 'its', 'we']
    if stop_words:
        out= all_title_words[~all_title_words.isin(stop_words)].value_counts().head(top_num)
    else:
        out= all_title_words.value_counts().head(top_num)

    return out

def plot_interests(date_col, interest_col):
    #plot interest rate between 1996 and 2023
    plt.rcParams['font.size'] = 16
    plt.figure(figsize=(12, 4))
    plt.plot(date_col, interest_col, color='blue', linewidth=2)
    plt.xlabel('Interest Rate Decision Date')
    plt.ylabel('Interest Rate (%)')
    plt.title('Federal Reserve Interest Rates by Date')
    #plt.xlim(1996, 2023)
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def styled_bar_plot(column, title='Interest Rate Decisions'):
    plt.figure(figsize=(4, 4))

    # Create the bar plot
    ax = column.value_counts().plot.bar(color='skyblue')

    # Set labels and title with increased pad value
    plt.title(title, pad=20)  # Increase the pad value here
    plt.xticks(rotation=0)  # Rotate x-axis labels for readability

    # Remove the y-axis
    ax.yaxis.set_visible(False)

    # Annotate each bar with its count
    for p in ax.patches:
        ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=12, color='black', xytext=(0, 5),
                    textcoords='offset points')

    # Adjust the distance from the top of the highest bar
    ax.set_ylim(0, column.value_counts().max() + 10)  # Adjust the value (e.g., +10) as needed

    plt.tight_layout()
    plt.show()
   
def speech_interest_visualize(year, speech_df, interest_df):
    #speeches speech_dates at 2019
    speech_year= speech_df['date'][speech_df['year']==year]

    interest_year= interest_df[['interest_date', 'decision_num']][interest_df['interest_date'].dt.year==year]

    fig, ax1 = plt.subplots(figsize=(15, 2))

    # Plot thin bars for each date (first plot)
    for date in speech_year:
        ax1.plot([date, date], [0, 1], color='b', linewidth=0.4)

    # Set the x-axis limits for the first plot
    ax1.set_xlim(speech_year.min(), speech_year.max())

    # Create a second y-axis (ax2) that shares the same x-axis
    ax2 = ax1.twinx()

    # Plot the scatter plot with thin bars for the second plot
    ax2.scatter(interest_year['interest_date'], interest_year['decision_num'], marker='2', color='r', s=100)

    # Set the x-axis labels from the second plot
    ax2.set_xticks(interest_year['interest_date'])
    ax2.set_xticklabels(interest_year['interest_date'].dt.strftime("%d-%m"), rotation=45)

    # Set the title for the combined plot
    plt.title(f'Speeches & Interest Rates in {year}', fontsize=16)

    # Set y-axis ticks for both subplots
    ax1.set_yticks([])

    # Set the y-axis tick labels to the decision values with labels
    decision_values= interest_df['decision_num'].unique()
    mapped_decision_values=np.array(interest_df['decision'].unique())
    ax2.set_yticks(decision_values)
    ax2.set_yticklabels(mapped_decision_values, fontsize=12)

    # Display the grid
    ax2.grid(True)

    # Create custom legend handles with desired line colors
    custom_legend_handles = [mlines.Line2D([], [], color='b', markersize=5, label='FED Speeches'),
                            mlines.Line2D([], [], color='r', marker='|', markersize=5, label='FED Interest Decision')]

    # Add a legend with custom legend handles
    ax1.legend(handles=custom_legend_handles, loc='upper left')
    ax2.yaxis.tick_left()

    plt.show()

#OLD TO DELETE LATER:
# import matplotlib.pyplot as plt
# import seaborn as sns
# 
# import matplotlib.ticker as ticker
# import nltk
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
# from wordcloud import WordCloud, STOPWORDS


#returns bar histograms for 9 features 
def ratings_bar_hist(data, explain_vars):
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    for i, column in enumerate(explain_vars):
        sns.countplot(data=data, x=column, palette = ["#97e3fc"], ax=axes[i]) #hue='OverallScore'
        axes[i].set_xlabel(column)
        axes[i].set_ylabel('Count')
        axes[i].set_title(f'Distribution of {column}')

    # Adjust the layout
    fig.tight_layout()

    # Show the plot
    plt.show()

#returns value counts of some features
    def vars_hist_table(data, rating_col):
        table = pd.DataFrame(columns=['rating', 'count'])
        for col in rating_col:
            value_counts = data[col].value_counts()
            table = table.append({'rating': col, 'count': value_counts.values[0]}, ignore_index=True)
        return table
    



#scatters sample prediction against real scores and marks tagged samples.
def scatter_tag_deviations(data,x_col,y_col,hue):
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col, hue=hue, palette='viridis', s=100, alpha=0.2, linewidth=0.2)

    ## Add a 45-degree line for reference
    plt.plot([data[x_col].min(), data[x_col].max()], [data[x_col].min(), data[x_col].max()], 
             color='red', linestyle='--', alpha=0.4, label='45-degree line')

    # Add labels and title
    plt.xlabel('Actual Values (OverallScore)')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')

    # Show the legend
    plt.legend(title='Reviews', loc='lower right', labels=['Mismatch', 'Match'])

    # Show the plot
    plt.grid(True)
    plt.show()



#compare distribution of scores in general population vs in when feature is null
def compare_score_dist(data, score_col, condition_col):
    df_null_review = data[[score_col]][data[condition_col].isnull()]
    df_all_reviews = data[[score_col]]

    # Create a shared axis for both plots
    fig, ax = plt.subplots(figsize=(8, 4))  # Adjust the figure size as needed

    # Plot both KDEs on the shared axis with bw_adjust
    sns.kdeplot(data=df_all_reviews[score_col], ax=ax, bw_method='scott', bw_adjust=2, label='All Reviews', color='blue')
    sns.kdeplot(data=df_null_review[score_col], ax=ax, label='Null Review', color='red')

    # Add labels and title
    ax.set_xlabel('Overall Score')
    ax.set_ylabel('Density')
    ax.set_title('KDE Plot of Overall Scores')

    # Set x-axis limits based on the minimum and maximum values in the 'OverallScore' column
    x_min = data[score_col].min()
    x_max = data[score_col].max()
    ax.set_xlim(x_min, x_max)

    # Show the legend
    ax.legend()

    # Show the plot
    plt.show()




#takes text and filters punctuation ans stop words
def punctuation_stop(text):
    """Remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered

#filter unwanted words list and patterns
def filter_words(words_filtered, unwanted_words):
    # Filter out words containing unwanted_words
    words_filtered = [word for word in words_filtered if word not in unwanted_words]

    # Filter out consecutive words with the same pattern
    filtered_words = [words_filtered[0]]
    for i in range(1, len(words_filtered)):
        if words_filtered[i] != words_filtered[i - 1]:
            filtered_words.append(words_filtered[i])

    return filtered_words

#Generate a word cloud from a DataFrame column of text.
def generate_word_cloud(df, column, unwanted_words, max_words=200):
    """
    Parameters:
        df (DataFrame): The DataFrame containing the text column.
        column (str): The name of the text column in the DataFrame.
        unwanted_words (list): List of unwanted words to filter from the cloud.
        max_words (int, optional): The maximum number of words in the word cloud. Default is 200.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    data_copy = df.copy()

    # Remove rows with NaN values in the specified column temporarily
    data_copy = data_copy.dropna(subset=[column])

    # Convert the 'Title' column to strings and join
    words = " ".join(data_copy[column].astype(str))

    words_filtered = punctuation_stop(words)

    filtered_words = filter_words(words_filtered, unwanted_words)

    # Join the filtered words back into a single text string
    text = " ".join(filtered_words)

    wc = WordCloud(background_color="white", random_state=1, stopwords=STOPWORDS, max_words=max_words, width=1500, height=800)
    wc.generate(text)

    plt.figure(figsize=[12, 10])
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()



#styled scatter
def styled_scatter(data, x, y):
    # Create a scatter plot
    data.plot.scatter(x, y, alpha=0.01)

    # Add labels and title
    plt.xlabel(f'{x}')
    plt.ylabel(f'{y}')
    plt.title(f'{x} vs. {y}')

    # Set y-axis ticks to show all score values
    score_ticks = data[y].unique()
    plt.yticks(score_ticks)

    # Limit x-axis to 800
    plt.xlim(0, 800)

    # Show the plot
    plt.tight_layout()
    plt.show()



def plot_one_class(data_col, date_data, text_num=None):
    date= date_data[text_num].strftime('%Y-%m-%d')
    plt.figure(figsize=(10, 3))
    plt.plot(data_col, linestyle='-', color='limegreen', label='Increase')
    plt.xlabel('Speech Progress (Number Of Sentences)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    if text_num>=0 & text_num<=date_data.count():
        plt.title(f'Interest Increase Probability - by {date} Speech Progress', fontsize=16)
    else:
        plt.title('Probability for Interest Increase - as Speech Progress', fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size': 10})
    plt.show()

#for some classes
def plot_some_classes(data, date_data, text_num=None):
    date= date_data[text_num].strftime('%Y-%m-%d')
    plt.figure(figsize=(10, 3))
    plt.plot(data.index, data.iloc[:,0], linestyle='-', color='tomato', label='Decrease', linewidth=2)
    plt.plot(data.index, data.iloc[:,1], linestyle='-', color='limegreen', label='Increase', linewidth=2)
    plt.plot(data.index, data.iloc[:,2], linestyle='-', color='slategrey', label='No change', linewidth=2)

    plt.xlabel('Speech Progress (Number Of Sentences)', fontsize=12)
    plt.ylabel('Probability', fontsize=12)
    plt.title('Probability For Interest Decision- by speech progress', fontsize=16)
    if text_num>=0 & text_num<=date_data.count():
        plt.title(f'Interest Increase Probability - by {date} Speech Progress', fontsize=16)
    else:
        plt.title('Interest Increase Probability- by speech progress', fontsize=16)
    plt.grid(True)
    plt.legend(loc='upper right', prop={'size': 9}) # Display legend
    plt.show()


#takes text and filters punctuation ans stop words
def punctuation_stop(text):
    """Remove punctuation and stop words"""
    filtered = []
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    for w in word_tokens:
        if w not in stop_words and w.isalpha():
            filtered.append(w.lower())
    return filtered

#filter unwanted words list and patterns
def filter_words(words_filtered, unwanted_words):
    # Filter out words containing unwanted_words
    words_filtered = [word for word in words_filtered if word not in unwanted_words]

    # Filter out consecutive words with the same pattern
    filtered_words = [words_filtered[0]]
    for i in range(1, len(words_filtered)):
        if words_filtered[i] != words_filtered[i - 1]:
            filtered_words.append(words_filtered[i])

    return filtered_words

#Generate a word cloud from a DataFrame column of text.
def generate_word_cloud(df, column, unwanted_words, max_words=200):
    """
    Parameters:
        df (DataFrame): The DataFrame containing the text column.
        column (str): The name of the text column in the DataFrame.
        unwanted_words (list): List of unwanted words to filter from the cloud.
        max_words (int, optional): The maximum number of words in the word cloud. Default is 200.
    """
    # Create a copy of the DataFrame to avoid modifying the original data
    data_copy = df.copy()

    # Remove rows with NaN values in the specified column temporarily
    data_copy = data_copy.dropna(subset=[column])

    # Convert the 'Texts' column to strings and join
    words = " ".join(data_copy[column].astype(str))

    words_filtered = punctuation_stop(words)

    filtered_words = filter_words(words_filtered, unwanted_words)

    # Join the filtered words back into a single text string
    text = " ".join(filtered_words)

    wc = WordCloud(background_color="white", random_state=1, stopwords=STOPWORDS, max_words=max_words, width=1500, height=800)
    wc.generate(text)

    plt.figure(figsize=[10, 10])
    plt.imshow(wc, interpolation="bilinear")
    plt.axis('off')
    plt.show()



def plot_kdes (df, dist_column, class_column, xlabel, title):
    # Get unique decision values
    class_types = df[class_column].unique()

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 5))

    # Iterate over each decision and plot KDE
    for class_type in class_types:
        subset_df = df[df[class_column] == class_type]
        subset_df[dist_column].plot(kind='kde', ax=ax, label=class_type)

    # Set labels and title
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title(title, fontsize=16)

    # Add legend
    ax.legend(title=class_column, fontsize=12)

    # Set x-axis limits to start from 0 and set ticks
    ax.set_xlim(left=0)

    # Customize tick parameters for better visibility
    ax.tick_params(axis='both', which='major', labelsize=12)

    # Show the plot
    plt.tight_layout()  # Adjust layout for better spacing
    plt.show()

