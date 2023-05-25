# This script classifies movie reviews as either positive or negative
# It uses VADER and Roberta, and prints percentage of correct classifications for each model
# The output of each model is not a classification, but contains different types of scores
# From these scores, one can then infer whether or not the review was positive or not

# We are using the labeled dataset 'imdb' from huggingface: https://huggingface.co/datasets/imdb

# Import libaries and load the dataset
import pandas as pd
import datasets

dataset = datasets.load_dataset('imdb')
df = pd.DataFrame(dataset['train'])

# This column is for the predictions and will be overwritten later
df.insert(0, 'ID', range(1, len(df) + 1))

## At row 12500 the labeled data switches from 0 to 1 in value
# df = df.iloc[12250:12750];

## If vader_lexicon has not been used before it will have to be downloaded
# import nltk
# nltk.download('vader_lexicon')

######################## Predictions using VADER #####################################

from nltk.sentiment import SentimentIntensityAnalyzer
from tqdm import tqdm
#Create Vaders sentiment-analyser
sia = SentimentIntensityAnalyzer()

# Run the polarity score on the entire dataset
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    text = row['text']
    myid = row['ID']
    res[myid] = sia.polarity_scores(text)

vaders = pd.DataFrame(res).T

# Rename column to enable join with original dataframe
vaders = vaders.reset_index().rename(columns={'index': 'ID'})
vaders = vaders.merge(df, how='left')

vaders['predictictedLabel'] = 0

## Using the compound value of 0 as a threshold for a positive prediction or negative prediction
for i, row in vaders.iterrows():
    if vaders.loc[i, 'compound'] > 0:
        vaders.loc[i, 'predictictedLabel'] = 1
    else:
        vaders.loc[i, 'predictictedLabel'] = 0

comparison_result = vaders['label'] == vaders['predictictedLabel']
print(comparison_result.mean() * 100)

######################## Predictions using Roberta #####################################

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

#This model is trained on twitter data, ideally you would train your own model
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# Method to calculate roberta values for each row
def polarity_scores_roberta(example):
    encoded_text = tokenizer(example, return_tensors='pt')
    output = model(**encoded_text)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    scores_dict = {
        'roberta_neg': scores[0],
        'roberta_neu': scores[1],
        'roberta_pos': scores[2]
    }
    return scores_dict

res = {}

# Get classifications for each row using the roberta method
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['text']
        myid = row['ID']
        vader_result_rename = {}
        roberta_result = polarity_scores_roberta(text)
        both = {**roberta_result}
        res[myid] = both
    except RuntimeError:
        print(f'Broke for id {myid}')

results_df = pd.DataFrame(res).T
# Rename column to enable join with original dataframe
results_df = results_df.reset_index().rename(columns={'index': 'ID'})
results_df = results_df.merge(df, how='left')

results_df['predictictedLabel'] = 0

# For the classification of the roberta output, the script looks at the positive value and the negative values,
# and whichever is largest determines if review was positive or negative
for i, row in results_df.iterrows():
    if results_df.loc[i, 'roberta_pos'] > results_df.loc[i, 'roberta_neg']:
        results_df.loc[i, 'predictictedLabel'] = 1
    else:
        results_df.loc[i, 'predictictedLabel'] = 0

comparison_result = results_df['label'] == results_df['predictictedLabel']
print(comparison_result.mean() * 100)

print("Done")
