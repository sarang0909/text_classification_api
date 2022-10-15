# text_classification_api : Production Ready Code


## About  
This is a project developed to create a code template and to understand different text classification techniques. This project includes different training notebooks to create different kind of text classification models. This project also includes a code to make productionaized text classification api using standard practices in MLOps.   
The models are developed on very small data.     


### NLP and MLOps techniques to learn/use from this end to end project:
1. collect,clean,annotate text data   
2. implement different methods of text classification models    
3. build inference api   
4. create streamlit application    
5. write unit test cases and performance test cases
6. code documentation
7. code formatting 
8. code deployment using docker and circleci
9. pre-commit hooks



This code can be used for end to end text classification project development as well as deployment.  
 
If you are only looking to learn/use model building techniques,directly jump to notebooks:   
1.[Text Classification using TF-IDF and Pycaret](src/training/tfidf_pycaret.ipynb)    
2.[Text Classification using TF-IDF and custom Machine Learning](src/training/tfidf_custom_ml.ipynb)    
3.[Text Classification using TF-IDF and custom Neural network using Keras](src/training/tfidf_custom_dl_keras.ipynb)  
4.[Text Classification using distilbert embeddings and custom Machine Learning](src/training/embedding_custom_ml.ipynb)    
5.[Text Classification using distilbert embeddings and Neural network using Pytorch](src/training/embedding_custom_dl.ipynb)    
6.[Text Classification using distilbert embeddings and Neural Network using huggging face trainer api](src/training/embedding_hugging_face.ipynb)    
7.[Text Classification using sentence transformer embeddings and custom neural network using Pytorch](src/training/embedding_sentence_transformer_custom_dl.ipynb)    


The basic code template for this project is derived from my another repo <a href="https://github.com/sarang0909/Code_Template">code template</a> 

The project considers following phases in ML project development lifecycle:  
Requirement    
Data Collection   
Model Building   
Inference   
Testing     
Deployment   

We have not considered model evaluation and monitoring.   


### Requirement

Create a text classification api which accepts a news article or a sentence from news article and classifies it into POSITIVE, NEGATIVE or NEUTRAL sentiment.   

### Data Collection   

News article data is collected by refering my another repo <a href="https://github.com/sarang0909/news_api">news api</a>. 

Then sample 100 sentences annotated using <a href="https://github.com/sarang0909/news_api">doccano</a>,a data annotation tool. Please note that since this is just a demo project,we have not used huge data. We have used only 100 sentences. In reality,data might be huge and any other data annotation technique can be used.  

### Model Building   
 
Input vectors used : TF-IDF, word embeddings from distilbert,word embeddings from sentence transformer

Model techniques used : Custom ML using sklearn,Custom Neural Network using keras,Custom Neural Network using pytorch, Custom Neural Network using hugging face transformers trainer api

Using combinations of above we have created 7 different models:   
  
 

| Input vectors | Model | Library| Model_Name
| --- | --- |--- |--- |
| TF-IDF | Best model from pycaret |Pycaret|tfidf_pycaret
| TF-IDF | ML model by experiments |sklearn|tfidf_custom_ml
| TF-IDF | Custom neural network |Keras|tfidf_custom_dl_keras
| Distilbert embeddings | ML model by experiments |sklearn,transformers|embedding_custom_ml
| Distilbert embeddings | Custom neural network |Pytorch,transformers|embedding_custom_dl
| Distilbert embeddings | transformers neural network|Pytorch,transformers|embedding_hugging_face
| sentence transformer embeddings | Custom neural network |Pytorch,sentence_transformer|embedding_sentence_transformer_custom_dl

You can try different combinations by creating/updating training notebooks of these models.



### Inference   
There are 2 ways to deploy this application.   
1. API using FastAPI.
2. Streamlit application

### Testing     
Unit test cases are written   

### Deployment 
Deployment is done locally using docker.   


## Code Oraganization   
Like any production code,this code is organized in following way:   
1. Keep all Requirement gathering documents in docs folder.       
2. Keep Data Collection and exploration notebooks  in src/training folder.  data_cleaning.ipynb, data_collection_eda.ipynb
3. Keep datasets in data folder.    
Raw data kept in raw_data csv.
Cleaned paragraphs stored in paragraph_clean_data.csv    
Cleaned sentencesstored in sentences_clean_data.csv   
Actual training done on 100_sentiment_analysis_sentences.csv
4. Keep model building notebooks at src/training folder.      
5. Keep generated model files at src/models.  
6. Write and keep inference code in src/inference.   
7. Write Logging and configuration code in src/utility.      
8. Write unit test cases in tests folder.<a href="https://docs.pytest.org/en/7.1.x/">pytest</a>,<a href="https://pytest-cov.readthedocs.io/en/latest/readme.html">pytest-cov</a>    
9. Write performance test cases in tests folder.<a href="https://locust.io/">locust</a>     
10. Build docker image.<a href="https://www.docker.com/">Docker</a>  
11. Use and configure code formatter.<a href="https://black.readthedocs.io/en/stable/">black</a>     
12. Use and configure code linter.<a href="https://pylint.pycqa.org/en/latest/">pylint</a>     
13. Add Git Pre-commit hooks.     
14. Use Circle Ci for CI/CD.<a href="https://circleci.com/developer">Circlci</a>    
 
Clone this repo locally and add/update/delete as per your requirement.   
Since we have used different design patterns like singleton,factory.It is easy to add/remove model to this code. You can remove code files for all models except the model which you want to keep as a final.   
Please note that this template is in no way complete or the best way for your project structure.   
This template is just to get you started quickly with almost all basic phases covered in creating production ready code.   

## Project Organization


├── README.md         		<- top-level README for developers using this project.    
├── pyproject.toml         		<- black code formatting configurations.    
├── .dockerignore         		<- Files to be ognored in docker image creation.    
├── .gitignore         		<- Files to be ignored in git check in.    
├── .pre-commit-config.yaml         		<- Things to check before git commit.    
├── .circleci/config.yml         		<- Circleci configurations       
├── .pylintrc         		<- Pylint code linting configurations.    
├── Dockerfile         		<- A file to create docker image.    
├── environment.yml 	    <- stores all the dependencies of this project    
├── main.py 	    <- A main file to run API server.    
├── main_streamlit.py 	    <- A main file to run API server.  
├── src                     <- Source code files to be used by project.    
│       ├── inference 	        <- model output generator code   
│       ├── model	        <- model files   
│       ├── training 	        <- model training code  
│       ├── utility	        <- contains utility  and constant modules.   
├── logs                    <- log file path   
├── config                  <- config file path   
├── data              <- datasets files   
├── docs               <- documents from requirement,team collabaroation etc.   
├── tests               <- unit and performancetest cases files.   
│       ├── cov_html 	        <- Unit test cases coverage report    

## Installation
Development Environment used to create this project:  
Operating System: Windows 10 Home  

### Softwares
Anaconda:4.8.5  <a href="https://docs.anaconda.com/anaconda/install/windows/">Anaconda installation</a>   
 

### Python libraries:
Go to location of environment.yml file and run:  
```
conda env create -f environment.yml
```

 

## Usage
Here we have created ML inference on FastAPI server with dummy model output.

1. Go inside 'text_classification_api' folder on command line.  
   Run:
  ``` 
      conda activate text_classification_api  
      python main.py       
  ```
  Open 'http://localhost:5000/docs' in a browser.
![alt text](docs/fastapi_first.jpg?raw=true)
![alt text](docs/fastapi_second.jpg?raw=true)
 
2. Or to start Streamlit application  
5. Run:
  ``` 
      conda activate text_classification_api  
      streamlit run main_streamlit.py 
  ```  
![alt text](docs/streamlit_first.jpg?raw=true)
![alt text](docs/streamlit_second.jpg?raw=true)
 
### Unit Testing
1. Go inside 'tests' folder on command line.
2. Run:
  ``` 
      pytest -vv 
      pytest --cov-report html:tests/cov_html --cov=src tests/ 
  ```
 
### Performance Testing
1. Open 2 terminals and start main application in one terminal  
  ``` 
      python main.py 
  ```

2. In second terminal,Go inside 'tests' folder on command line.
3. Run:
  ``` 
      locust -f locust_test.py  
  ```

### Black- Code formatter
1. Go inside 'text_classification_api' folder on command line.
2. Run:
  ``` 
      black src 
  ```

### Pylint -  Code Linting
1. Go inside 'text_classification_api' folder on command line.
2. Run:
  ``` 
      pylint src  
  ```

### Containerization
1. Go inside 'text_classification_api' folder on command line.
2. Run:
  ``` 
      docker build -t myimage .  
      docker run -d --name mycontainer -p 5000:5000 myimage         
  ```

### Pre-commit hooks
1. Go inside 'text_classification_api' folder on command line.
2. Run:
  ``` 
      pre-commit install  
  ```
3. Whenever the command git commit is run, the pre-commit hooks will automatically be applied.     
4. To test before commit,run:  

  ``` 
      pre-commit  run 
  ```    

### CI/CD using Circleci
1. Add project on circleci website then monitor build on every commit.



## Note
1.embedding_custom_dl,embedding_hugging_face models are not checked in because of size,you can generate them by running corresponding training notebook.   
2.You'll need to create news api key to get news data,so create and update api key in data_collection notebook.       


## Contributing
Please create a Pull request for any change. 

## License


NOTE: This software depends on other packages that are licensed under different open source licenses.

