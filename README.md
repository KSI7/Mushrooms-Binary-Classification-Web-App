# Binary-Classification-Web-App
A Binary Classification Web App which classifies hypothetical samples of 23 species of gilled mushrooms into definitely edible, definitely poisonous or of unknown edibility and not recommended. This latter class was combined with the poisonous one..

Three classifiers are used to classify the mushrooms - Support Vector Machine (SVM), Logistic Regression and Random Forest Classifier. You can choose any one of these.

Results can be plotted using three metrics - Confusion Matrix, Reciever Operating Charecteristic (ROC) Curve and Precision-Recall Curve. The results show the model's accuracy, precision and recall values.

Different Model Hyperparameters have been included with different classifiers with varying results.

To run -

1. Clone the repository.
2. Open the app.py file using either an IDE or a text editor and set the path to the mushrooms.csv dataset (Line 20).
3. Open Command Prompt and change the directory to that of the clone using the cd command.
4. To host the web app on the localhost, type the following in Command Prompt - streamlit run app.py
5. Your web app will be successfully hosted on the localhost.

Libraries used -

1. Streamlit - 0.62.0
2. Scikit-learn - 0.23.1
3. Pandas - 1.0.5
4. Numpy - 1.19.0

Python Version - 3.7.7
