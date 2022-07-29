# Mushrooms-Binary-Classification-Web-App
A Binary Classification Web App which classifies hypothetical samples of 23 species of gilled mushrooms into definitely edible, definitely poisonous or of unknown edibility and not recommended. This latter class was combined with the poisonous one. The results show the model's accuracy, precision and recall values.

### Classifiers 
1. Support Vector Machine (SVM)
2. Logistic Regression 
3. Random Forest Classifier

### Metrics 
1. Confusion Matrix
2. Reciever Operating Charecteristic (ROC) Curve 
3. Precision-Recall Curve 

Different Model Hyperparameters have been included with different classifiers with varying results.

### [Deployment Link](https://ksi7-binary-classification-web-app-app-8prmpv.streamlitapp.com/)

### To run - (locally)
1. Clone the repository.
2. Open the app.py file using either an IDE or a text editor and set the path to the mushrooms.csv dataset (Line 20).
3. Open Command Prompt and change the directory to that of the clone using the **cd** command.
4. Run the following command to install all dependencies - 
 > **python pip install -r requirements.txt**
5. To host the web app on the localhost, type the following in Command Prompt or Windows Powershell (recommended) -
 > **streamlit run app.py**
6. Your web app will be successfully hosted on the localhost.

### Libraries -

1. [Streamlit](https://www.streamlit.io/) - 0.62.0
2. [Scikit-learn](https://scikit-learn.org/stable/) - 0.23.1
3. [Pandas](https://pandas.pydata.org/) - 1.0.5
4. [Numpy](https://numpy.org/) - 1.19.0

[Python](https://www.python.org/downloads/release/python-377/) Version - 3.7.7
