import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    INPUT - 
    messages_filepath - Filepath of Disaster_messgaes.csv 
    categories_filepath - Filepath of Disaster_category.csv
    
    OUTPUT - Returns the dataframe    
    """
    df_msg = pd.read_csv(messages_filepath)
    df_cat = pd.read_csv(categories_filepath)
    df = pd.merge(df_msg,df_cat,on="id")
    return df
    
def clean_data(df):
    """
    Clean Data :
    1. Clean and Transform Category Columns from categories csv
    2.Drop Duplicates
    3.Remove any missing  values
    Args:
    
    INPUT - df - merged Dataframe from load_data function
    OUTPUT - Returns df - cleaned Dataframe
    """
        # Split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)
    row = categories.iloc[0]
    
    # Get new column names from category columns
    category_colnames = row.apply(lambda x: x.rstrip('- 0 1'))
    categories.columns = category_colnames
    
    # Convert category values to 0 or 1
    categories = categories.applymap(lambda s: int(s[-1]))

    # Drop the original categories column from Dataframe
    df.drop('categories', axis=1, inplace=True)
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df_final = pd.concat([df, categories], axis=1)
    
    #Drop missing values and duplicates from the dataframe
    
    df_final.drop_duplicates(subset='message', inplace=True)
    df_final.dropna(subset=category_colnames, inplace=True)
    
    #Refer ETL Pipeline preparation Notebook to understand why these columns are dropped
    df_final = df_final[df_final.related != 2]
    df_final = df_final.drop('child_alone', axis=1)
    
    return df_final

    
def save_data(df, database_filename):
    """
    Load the clean dataset into sqlite database
    
    Args :
    df - cleaned dataframe from clean_data function()
    database_filename(string): the file path to save file .db
    
    """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster_messages', engine, index=False, if_exists='replace')
    engine.dispose()

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()