DISASTER RESPONSE PIPELINE PROJECT
------------------------------------
The three major aspects of this project is as follows:

* ETL Pipeline - Clean,Transform and load data to Sqlite database
* ML Pipeline - Build ML Pipeline 
* Flask web-app displaying analysis from data

A __web app__ is created with Flask and Bootstrap for Natural Language Processing (__NLP__). The app provides an __interface__ for new messages, (e.g. Twitter messages scanned by disaster relief agencies in a _Disaster Response_ situation). Whenever you type a message it is classified into 37 Categories based on the learnings from the trained dataset

File Structure
----------------------
* `requirements.txt`: contains the environment requirements to run program

* `data` folder contains the following:
    * `disaster_categories.csv`: contains the disaster categories csv file
    * `disaster_messages.csv`: contains the disaster messages csv file
    * `DisasterResponse.db`: contains the DisasterResponse db which is a merge of messages and categories by ID
    * `proccess_data.py`: contains the scripts to run etl pipeline for cleaning data
	
* `Jupyter Notebooks` folder contains the following:
    * `ETL Pipeline Preparation.ipynb`: contains the code developement process to create ETL Pipeline
    * `ML Pipeline Preparation.ipynb`: contains the code developement process to create ML Pipeline
    * `DisasterResponse.db`: contains the DisasterResponse db which is a merge of messages and categories by ID
	
* `ML model` folder contains the following:
    * `ml_pipeline.py`: contains scripts that create ml pipeline
    * `disaster_resonse.pkl`: contains the Classifier pickle fit file
    * `train_classifier.py`: script to train_classifier.py
	
* `app` folder contains the following:
  * `templates`: Folder containing
    * `index.html`: Renders homepage
    * `go.html`: Renders the message classifier
  * `run.py`: Defines the app routes
  
* `img` folder contains snapshots taken from web app:

INSTALLATION
----------------------
### Clone Repo

### Rerun Scripts
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
       - `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
	   
    - To run ML pipeline that trains classifier and saves
        - `python ML model/train_classifier.py data/DisasterResponse.db ML model/disaster_response.pkl`

2. Run the following command in the app's directory to run your web app.
    `python3 app/run.py

3. Go to http://0.0.0.0:3001/

RESULTS
---------------------------
Results are as follows:

1. The weighted avg precision and F1 score obtained from a test run using SGDClassifier are 0.75 and 0.61 respectively
2. The weighted avg precision and F1 score obtained from a test run using RandomForestClassifier 0.75 and 0.57 respectively
3. The weighted avg precision and F1 score obtained from a test run using GradientBoostingClassifier 0.70 and 0.60 respectively

Extending this
----------------------
If you want to extend this work, here are few places to start with

1. Use any other Classifier and check if the precision is improved
2. Include more graphs
3. Reduce the run time of the code by make use of less sample size instead of entire dataset


