#https://www.kaggle.com/code/ayushs9020/lung-cancer-prediction-99-98/input
#_____________________________Libraries import______________________________#
import tkinter as tk
from tkinter import messagebox
import pandas as pd
import csv
# **DATA PROCESSING**

import numpy as np # Array Processing
import pandas as pd # Data Processing 
import os # Input of Data

# **DATA ANALYSIS**
import matplotlib.pyplot as plt # Plots


# **Machine learning**
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,roc_auc_score,confusion_matrix
from sklearn.metrics import roc_curve,classification_report,matthews_corrcoef,precision_score
import seaborn as sns
from imblearn.over_sampling import RandomOverSampler

from sklearn.model_selection import train_test_split
#--------------------------Function code area---------------------------------#
#Functions area#
#Preprocessing data function
def pre(dataframe):
    #define a list name target
    target = ["LUNG_CANCER"]
    #drop the column "Level"
    x = dataframe.drop(target , axis = 1)
    #let the "level" has a own list called y
    y = dataframe[target]
    return x , y


def check_risk():
    
    # Load data from CSV file
    df = pd.read_csv("E:\Testing case.csv")

    # Get the last row of the dataframe
    last_row = df.tail(1)

    # Convert last row into a new dataframe
    input_data = last_row.drop(["LUNG_CANCER","INDEX","Patient Id"], axis=1)

    prediction=model.predict(input_data)

    if prediction == 0:
        prediction = "LOW"
    elif prediction == 1:
        prediction = "HIGH"
    
    
    # # Write the prediction to the CSV file
    df.loc[df.index[-1], "LUNG_CANCER"] = prediction
    df.to_csv("E:\Testing case.csv", index=False)
    
    messagebox.showinfo("Risk Prediction", f"The predicted risk is {prediction}")


def submit_ic():
    ic = ic_entry.get()
    start_test_window(ic)
    
def check_Result():
    ic_number = ic_entry.get()

    try:
        df = pd.read_csv('E:\Testing case.csv')
        result = df.loc[df['Patient Id'] == int(ic_number)]
        if not result.empty:
            message = ''
            for col in result.columns:
                message += f'{col}: {result[col].values[0]}\n'
            messagebox.showinfo('Result', message)
        else:
            messagebox.showwarning('Result', 'No results found for this IC number.')
    except FileNotFoundError:
        messagebox.showerror('Error', 'Results file not found.')
    except Exception as e:
        messagebox.showerror('Error', str(e))

def input_data_window():
    new_window2 = tk.Tk()
    new_window2.geometry("300x500")
    new_window2.title("Data entered")
    df = pd.read_csv('E:\Testing case.csv')
    questions = df.columns.values
    df.drop(["LUNG_CANCER"],axis = 1,inplace = True)

    # get the last row of data
    last_row = df.iloc[-1].tolist()
    
    # add labels for each column in the last row
    tk.Label(new_window2, text="Index: ").grid(row=0, column=0)
    tk.Label(new_window2, text=last_row[0]).grid(row=0, column=1)
    
    tk.Label(new_window2, text="IC Number: ").grid(row=1, column=0)
    tk.Label(new_window2, text=last_row[1]).grid(row=1, column=1)
    
    tk.Label(new_window2, text="Age: ").grid(row=3, column=0)
    tk.Label(new_window2, text=last_row[3]).grid(row=3, column=1)
    
    tk.Label(new_window2, text="Gender: ").grid(row=2, column=0)
    tk.Label(new_window2, text=last_row[2]).grid(row=2, column=1)
    
    # add labels for each question
       # create table entries
    questions = df.columns.values
    num_questions = len(questions)
    
    for i in range(num_questions):
        tk.Label(new_window2, text=questions[i] + ": ").grid(row=i, column=0)
        tk.Label(new_window2, text=last_row[i]).grid(row=i, column=1)
    
    def close_and_reopen():
        new_window2.destroy()  # Close new_window2
        root.deiconify()  # Reopen root window
        
    tk.Button(new_window2, text="Yes",command = check_risk).grid(row=num_questions, column=0)
    back_button = tk.Button(new_window2, text="Back", command=close_and_reopen)
    back_button.grid(row=num_questions+1, column=0)
    
    
def return_to_main_menu(window):
    window.destroy()
    root.deiconify()

def check_and_remove():
    patient_id=int(ic_entry.get())
    # Load the CSV file into a DataFrame
    df = pd.read_csv('E:\Testing case.csv')
    
    if patient_id in df['Patient Id'].values:
        print(f"Element {patient_id} exists in the DataFrame.")
    
        df = df[df['Patient Id'] != patient_id]

       # Overwrite the original CSV file with the updated DataFrame
        df.to_csv('E:\Testing case.csv', index=False)
        print("Data for patient ID", patient_id, "removed successfully.")
        
    else:
        print(f"Element {patient_id} does not exist in the DataFrame.")
    
    submit_ic()

#--------------------------Machine learning Train and Test ------------------------------#
df = pd.read_csv("E:\\survey lung cancer.csv")
# df.drop(["GENDER"], axis = 1 , inplace = True)
# Assuming you have a DataFrame named 'df'
df['GENDER'] = df['GENDER'].replace({"M": 1, "F": 0})

#replacing the dataset's patient risk to 0,1,2
df.replace(to_replace = "NO" , value = int(0) , inplace = True)
df.replace(to_replace = "YES" , value = 1 , inplace = True)

#a as dataset case variables, b as dataset patient's result
a = df.drop("LUNG_CANCER" , axis = 1)
b = df["LUNG_CANCER"]

over_samp =  RandomOverSampler(random_state=0)
X_train_ros, y_train_ros = over_samp.fit_resample(a, b)
X_train_ros.shape, y_train_ros.shape

X_train, X_test, y_train, y_test= train_test_split(X_train_ros, y_train_ros, test_size = 0.3, random_state = 42)

print(X_train.shape)
print(X_test.shape)

#import the Random Forest Classifier
model = RandomForestClassifier()
model.fit(X_train , y_train.values.ravel())


# predict the test set
Y_pred = model.predict(X_test)

# calculate the F1 score
f1 = f1_score(y_test, Y_pred, average='macro')

print("F1 score: {:.2f}".format(f1))
value=pd.DataFrame({'Actual Value':y_test,'Predicted Value':Y_pred})



# predict the test set
Y_pred = model.predict(X_test)

# calculate the confusion matrix
cm = confusion_matrix(y_test, Y_pred)


# Extract the true positive (TP) and false negative (FN) values
TP = cm[1, 1]
FN = cm[1, 0]

# Calculate sensitivity
sensitivity = TP / (TP + FN)

print("Sensitivity: {:.2f}".format(sensitivity))


# Calculate precision
precision = precision_score(y_test, Y_pred)

print("Precision: {:.2f}".format(precision))


# Calculate AUC score for SVM Classifier
auc_rf = roc_auc_score(y_test, Y_pred)
print("AUC Score: {:.2f}".format(auc_rf))
# calculate AUC score
fpr, tpr, _ = roc_curve(y_test, Y_pred)
# plot the ROC curve
plt.figure()
plt.plot(fpr, tpr, color='Red', lw=2, label='(AUC = {:.2f})'.format(auc_rf))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()
cn=df.corr()
print(cn)

# Calculate the MCC
mcc = matthews_corrcoef(y_test, Y_pred)

# Print the MCC score
print("MCC:", mcc)

#Correlation 
cmap=sns.diverging_palette(260,-10,s=50, l=75, n=6,
as_cmap=True)
plt.subplots(figsize=(18,18))
sns.heatmap(cn,cmap="coolwarm",annot=True, square=True,annot_kws={"fontsize": 14})

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()
#Model accuracy
rf_cr=classification_report(y_test, Y_pred)
print(rf_cr)
# Get feature importances
importances = model.feature_importances_

# Create a list of feature names
feature_names = X_train.columns

# Create a DataFrame to store feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the features by importance in descending order
feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)

# Print the feature importance rankings
print(feature_importance_df)

#______________________________________________________________________________#
    
def start_test_window(ic):
    root.withdraw()
    # create new window
    new_window = tk.Tk()
    new_window.geometry("300x600")
    new_window.title("Lung Cancer Risk Prediction Test")
    
    # add widgets to new window
    tk.Label(new_window, text="Your IC number: ").grid(row=0,column=0)
    tk.Label(new_window, text=ic).grid(row=0,column=1)
    
    # create table labels
    tk.Label(new_window, text="Question").grid(row=1, column=0)
    tk.Label(new_window, text="Answer").grid(row=1, column=1)
    
    df = pd.read_csv("E:\Testing case.csv")
    df.drop(["INDEX","Patient Id","GENDER","AGE","LUNG_CANCER"],axis = 1,inplace = True)
    # create table entries
    questions = df.columns.values
    
    gender_var = tk.StringVar(new_window)
    gender_var.set("Select Gender")
    gender_choices = ["Male", "Female"]

    
    age_var = tk.StringVar(new_window)
    #yes=1 no =0
    level_choices = ["0", "1"]
    
    level_var = []
    for i in range(2, len(questions)+2):
        var = tk.StringVar(new_window)
        var.set("Select Category")
        level_var.append(var)
    # create table
    for i in range(len(questions)+3):
        if i<=14:
            if i == 0:
               tk.Label(new_window, text="Age").grid(row=i+2, column=0)
               tk.Entry(new_window, textvariable=age_var).grid(row=i+2, column=1)
            elif i == 1:
                tk.Label(new_window, text="Gender").grid(row=i+2, column=0)
                tk.OptionMenu(new_window,  gender_var, *gender_choices).grid(row=i+2, column=1)
            else:
                tk.Label(new_window, text=questions[i-2]).grid(row=i+2, column=0)
                tk.OptionMenu(new_window, level_var[i-2], *level_choices).grid(row=i+2, column=1)
    def submit():
        with open("E:\Testing case.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            if file.tell() == 0:
                writer.writerow(['Index', 'IC Number', 'Age', 'Gender'] + ['Question ' + str(i+1) for i in range(len(level_var))])
            
        with open("E:\Testing case.csv", mode='r', newline='') as file:
            index = sum(1 for _ in csv.reader(file))
        
        with open("E:\Testing case.csv", mode='a', newline='') as file:
            writer = csv.writer(file)
            gender=gender_var.get()
            if(gender == "Male"):
                gender = 1
            else:
                gender = 0
            answers = [index, ic,gender, age_var.get() ] + [level_var[i].get() for i in range(len(level_var))]
            writer.writerow(answers)
            print("Submission successful. Index: ", index)
            
        new_window.destroy()
        input_data_window()

    # create submit button
    tk.Button(new_window, text="Send out",command=submit).grid(row=18, column=1)

# create main window
root = tk.Tk()
root.geometry("400x200")
root.title("Lung Cancer Risk Prediction Model")

# create IC entry box
ic_label = tk.Label(root, text="Enter IC number:")
ic_label.pack()
ic_entry = tk.Entry(root)
ic_entry.pack()

# create submit for test button
submit_button = tk.Button(root, text="Start Random Forest Test", command=check_and_remove)
submit_button.pack()

# create submit for test button
checkResultButton = tk.Button(root, text="Check Result", command=check_Result)
checkResultButton.pack()

close_button = tk.Button(root, text="Close", command=root.destroy)
close_button.pack()
root.mainloop()
