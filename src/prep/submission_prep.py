#Importing the necessary libraries
import pandas as pd
import regex as re

#Importing specific modules from the Natural Language Toolkit (NLTK) library
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#Load the 'prompts_train.csv' file into a DataFrame named 'prompts_df'
prompts_df = pd.read_csv('prompts_train.csv')

#Load the 'summaries_train.csv' file into a DataFrame named 'summaries_df'
summaries_df = pd.read_csv('summaries_train.csv')

#Create a set of stopwords (commonly occurring words that are usually ignored in text analysis)
stopwords = set(stopwords.words('english'))

#Initialize a WordNetLemmatizer, which is used to lemmatize words (reduce them to their base or dictionary form)
lemmatizer = WordNetLemmatizer()



# Define a function to lemmatize a string containing words
def lemmatize_words(string):
    # Split the input string into individual words
    words = string.split()
    # Lemmatize each word in the list using the lemmatizer
    words = [lemmatizer.lemmatize(word,pos='v') for word in words]
    # Join the lemmatized words back into a string
    return ' '.join(words)


# Define a function to clean text data in a DataFrame
def clean_text(df, column, column_new):
    """Removes non-alphabetic characters, converts to lowercase, removes stopwords, applies lemmatizer."""
    # Remove any characters that are not alphabetic letters from the specified column
    df[column_new] = df[column].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
    # Convert all text in the cleaned column to lowercase
    df[column_new] = df[column_new].apply(lambda x: x.lower())
    # Remove stopwords from the cleaned text by splitting, filtering, and rejoining words
    df[column_new] = df[column_new].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopwords)]))
    # Apply the lemmatization function to the cleaned text in the column
    df[column_new] = df[column_new].apply(lemmatize_words)
    # Return the DataFrame with the cleaned and processed text
    return df


# Clean the text in the 'prompt_text' column of 'prompts_df', creating a new column 'cleaned_prompt_text'
prompts_df = clean_text(prompts_df, 'prompt_text', 'cleaned_prompt_text')

# Print the cleaned text of the first prompt in the DataFrame
print(prompts_df['cleaned_prompt_text'][0])

