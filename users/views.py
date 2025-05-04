from django.shortcuts import render
from django.conf import settings
from django.http import HttpResponse
from django.contrib import messages
from .forms import UserRegistrationForm
from .models import UserRegistrationModel

import pandas as pd
import numpy as np

# Create your views here.




def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method =='POST':
        loginid=request.POST.get('loginid')
        pswd=request.POST.get('pswd')
        print("Login ID = ", loginid, ' Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(
                loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/userhome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'userlogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
            messages.success(request, 'Invalid Login id and password')
        return render(request, 'userlogin.html', {})


def UserHome(request):

    return render(request, 'users/userhome.html', {})

def viewData(request):
    import pandas as pd
    from django.conf import settings
    import os
    path=os.path.join(settings.MEDIA_ROOT,'chicago_crime_2014.csv')
    df=pd.read_csv(path)
    df=df.to_html()
    # path = os.path.join(settings.MEDIA_ROOT,'chicago_crime_2015.csv')
    # auto_df = pd .read_csv(path)
    # auto_df = auto_df.to_html
    # path = os.path.join(settings.MEDIA_ROOT,'chicago_crime_2016.csv')
    # auto_df1 = pd .read_csv(path)
    # auto_df1 = auto_df1.to_html
    return render(request, 'users/userviewdata.html', {'data': df})



def crime_training(request):
    import matplotlib
    import matplotlib.pyplot as plt
    import seaborn as sns

    #Preprocessing Libraries
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, accuracy_score, f1_score

    # ML Libraries
    from sklearn.ensemble import RandomForestClassifier 
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier

    # Evaluation Metrics
    from yellowbrick.classifier import ClassificationReport
    from sklearn import metrics
    import os
    # path=os.path.join(settings.MEDIA_ROOT+'//'+'chicago_crime_2014.csv')
    # path1=os.path.join(settings.MEDIA_ROOT+'//'+'chicago_crime_2015.csv')
    # path2=os.path.join(settings.MEDIA_ROOT+'//'+'chicago_crime_2016.csv')
    # df = pd.concat([pd.read_csv('D:\\my_projects\\Crime-Prediction-through-Machine-Learning-Algorithms\\media\\chicago_crime_2014.csv')], ignore_index=True)
    # df = pd.concat([df, pd.read_csv('D:\\my_projects\\Crime-Prediction-through-Machine-Learning-Algorithms\\media\\chicago_crime_2015.csv')], ignore_index=True)
    # df = pd.concat([df, pd.read_csv('D:\\my_projects\Crime-Prediction-through-Machine-Learning-Algorithms\\media\\chicago_crime_2016.csv')], ignore_index=True) 
    # df = pd.concat([pd.read_csv(path, error_bad_lines=False)], ignore_index=True)
    # df = pd.concat([df, pd.read_csv(path1, error_bad_lines=False)], ignore_index=True)
    # df = pd.concat([df, pd.read_csv(path2, error_bad_lines=False)], ignore_index=True)
    file_names = ['chicago_crime_2014.csv', 'chicago_crime_2015.csv', 'chicago_crime_2016.csv']
    dataframes = []
    for fname in file_names:
        path = os.path.join(settings.MEDIA_ROOT, fname)
        if os.path.exists(path):
            dataframes.append(pd.read_csv(path))
    df = pd.concat(dataframes, ignore_index=True)
    df = df.dropna()
    df = df.sample(n=100)
    
    df = df.drop(['ID', 'Case Number'], axis=1)
    df['Block'] = pd.factorize(df["Block"])[0]
    df['IUCR'] = pd.factorize(df["IUCR"])[0]
    df['Description'] = pd.factorize(df["Description"])[0]
    df['LocationDescription'] = pd.factorize(df["LocationDescription"])[0]
    df['FBICode'] = pd.factorize(df["FBICode"])[0]
    df['Target'] = df['Arrest'].replace([True,False],[1,0])
    Target = 'PrimaryType'
    print('Target: ', Target)
    plt.figure(figsize=(14,10))
    plt.title('Amount of Crimes by PrimaryType')
    plt.ylabel('Crime Type')
    plt.xlabel('Amount of Crimes')

    df.groupby([df['PrimaryType']]).size().sort_values(ascending=True).plot(kind='barh')

    # plt.show()
    # First, we sum up the amount of Crime Type happened and select the last 13 classes
    all_classes = df.groupby(['PrimaryType'])['Block'].size().reset_index()
    all_classes['Amt'] = all_classes['Block']
    all_classes = all_classes.drop(['Block'], axis=1)
    all_classes = all_classes.sort_values(['Amt'], ascending=[False])
    # After that, we replaced it with label 'OTHERS'
    df.loc[df['PrimaryType'].isin(['PrimaryType']), 'PrimaryType'] = 'OTHERS'

    # Plot Bar Chart visualize Primary Types
   
  

    df.groupby([df['PrimaryType']]).size().sort_values(ascending=True).plot(kind='barh')

   
    Classes = df['PrimaryType'].unique()
    df['PrimaryType'] = pd.factorize(df["PrimaryType"])[0]
    df['PrimaryType'].unique()
    X_fs = df.drop(['PrimaryType'], axis=1)
    Y_fs = df['PrimaryType']

    # Assuming 'Date' is a datetime column in your dataset
# # Convert 'Date' column to numerical features
#     df['Year'] = df['Date'].dt.year
#     df['Month'] = df['Date'].dt.month
#     df['Day'] = df['Date'].dt.day
#     df['Hour'] = df['Date'].dt.hour

# Drop the original 'Date' column
    df = df.drop('Date', axis=1)


    #Using Pearson Correlation
    plt.figure(figsize=(20,10))
    cor = df.corr()
    sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
    plt.show()
    Features = ["IUCR", "Description", "FBICode"]
    print('Full Features: ', Features)
    #Split dataset to Training Set & Test Set
    x, y = train_test_split(df,
                            test_size = 0.2,
                            train_size = 0.8,
                            random_state= 3)

    x1 = x[Features]    #Features to train
    x2 = x[Target]      #Target Class to train
    y1 = y[Features]    #Features to test
    y2 = y[Target]      #Target Class to test

    print('Feature Set Used    : ', Features)
    print('Target Class        : ', Target)
    print('Training Set Size   : ', x.shape)
    print('Test Set Size       : ', y.shape)
    # Create Model with configuration
    rf_model = RandomForestClassifier(n_estimators=70, # Number of trees
                                    min_samples_split = 30,
                                    bootstrap = True,
                                    max_depth = 50,
                                    min_samples_leaf = 25)

    # Model Training
    rf_model.fit(X=x1,
                y=x2)

    # Prediction
    result = rf_model.predict(y[Features])
    # Model Evaluation
    ac_sc = accuracy_score(y2, result)
    rc_sc = recall_score(y2, result, average="weighted")
    pr_sc = precision_score(y2, result, average="weighted")
    f1_sc = f1_score(y2, result, average='micro')
    confusion_m = confusion_matrix(y2, result)
    print("========== RandomForestClassifier Results ==========")
    print("Accuracy    : ", ac_sc)
    print("Recall      : ", rc_sc)
    print("Precision   : ", pr_sc)
    print("F1_Score    : ", f1_sc)
    print("Confusion Matrix: ")
    print(confusion_m)
    target_names = Classes
    visualizer = ClassificationReport(rf_model, classes=target_names)
    visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
    visualizer.score(y1, y2)       # Evaluate the model on the test data

    

    # g = visualizer.poof()
    # Create Model with configuration
    nn_model = MLPClassifier(solver='adam',
                            alpha=1e-5,
                            hidden_layer_sizes=(40,),
                            random_state=1,
                            max_iter=1000
                            )

    # Model Training
    nn_model.fit(X=x1,
                y=x2)

    # Prediction
    result = nn_model.predict(y[Features])
    ac_sc1 = accuracy_score(y2, result)
    rc_sc1 = recall_score(y2, result, average="weighted")
    pr_sc1 = precision_score(y2, result, average="weighted")
    f1_sc1 = f1_score(y2, result, average='micro')
    confusion_m = confusion_matrix(y2, result)
    print("========== Neural Network Results ==========")
    print("Accuracy1    : ", ac_sc1)
    print("Recall1      : ", rc_sc1)
    print("Precision1   : ", pr_sc1)
    print("F1_Score1    : ", f1_sc1)
    print("Confusion Matrix: ")
    print(confusion_m)
    target_names = Classes
    visualizer = ClassificationReport(nn_model, classes=target_names)
    visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
    visualizer.score(y1, y2)
    # Create Model with configuration
    knn_model = KNeighborsClassifier(n_neighbors=3)

    # Model Training
    knn_model.fit(X=x1,
                y=x2)

    # Prediction
    result = knn_model.predict(y[Features])
    ac_sc2 = accuracy_score(y2, result)
    rc_sc2 = recall_score(y2, result, average="weighted")
    pr_sc2 = precision_score(y2, result, average="weighted")
    f1_sc2 = f1_score(y2, result, average='micro')
    confusion_m = confusion_matrix(y2, result)

    print("========== K-Nearest Neighbors Results ==========")
    print("Accuracy2    : ", ac_sc2)
    print("Recall2      : ", rc_sc2)
    print("Precision2   : ", pr_sc2)
    print("F1_Score2    : ", f1_sc2)
    print("Confusion Matrix: ")
    print(confusion_m)
    target_names = Classes
    visualizer = ClassificationReport(knn_model, classes=target_names)
    visualizer.fit(X=x1, y=x2)     # Fit the training data to the visualizer
    visualizer.score(y1, y2)       # Evaluate the model on the test data

    print('================= Classification Report =================')
    print('')
    # print(classification_report(y2, result, target_names=target_names))

    # g = visualizer.poof()
    return render(request,'users/crime.html',{'Accuracy':ac_sc,'Recall':rc_sc,'Precision':pr_sc
                                             ,'F1_Score':f1_sc,'Accuracy1':ac_sc1,'Recall1':rc_sc1,'Precision1':pr_sc1
                                             ,'F1_Score1':f1_sc1,'Accuracy2':ac_sc2,'Recall2':rc_sc2,'Precision2':pr_sc2
                                             ,'F1_Score2':f1_sc2})
def crimeprediction(request):
    if request.method == 'POST':
        import os
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        PrimaryType = request.POST.get("PrimaryType")
        LocationDescription = request.POST.get("LocationDescription")
        District = request.POST.get("District")
        Ward = request.POST.get("Ward")
        CommunityArea = request.POST.get("CommunityArea")
        FBICode = request.POST.get("FBICode")
        Latitude = float(request.POST.get("Latitude"))
        Longitude = float(request.POST.get("Longitude"))
        # path=os.path.join(settings.MEDIA_ROOT+'//'+'chicago_crime_2014.csv')
        path=settings.MEDIA_ROOT+'//'+'chicago_crime_2014.csv'
        path1=os.path.join(settings.MEDIA_ROOT,'chicago_crime_2015.csv')
        path2=os.path.join(settings.MEDIA_ROOT,'chicago_crime_2016.csv')
        df = pd.concat([pd.read_csv(path)], ignore_index=True)
        df = pd.concat([df, pd.read_csv(path1)], ignore_index=True)
        df = pd.concat([df, pd.read_csv(path2)], ignore_index=True)    
        #df = pd.concat([pd.read_csv(path, error_bad_lines=False)], ignore_index=True)
        #df = pd.concat([df, pd.read_csv(path1, error_bad_lines=False)], ignore_index=True)
        #df = pd.concat([df, pd.read_csv(path2, error_bad_lines=False)], ignore_index=True)
        df = df.dropna()
        df = df.sample(n=1000)
        df = df.drop(['ID'], axis=1)
        df['LocationDescription'] = pd.factorize(df["LocationDescription"])[0]
        df['FBICode'] = pd.factorize(df["FBICode"])[0]
        df['PrimaryType'] = pd.factorize(df["PrimaryType"])[0]

        df['Target'] = df['Arrest'].replace([True,False],[1,0])
        df = df.drop(['Case Number','Date','Block','IUCR','Arrest','Description','Domestic','Beat'], axis=1)

        x=df.drop(['Target'],axis=1)
        y = df['Target'] 

        test_features = [PrimaryType,LocationDescription,District,Ward,CommunityArea,
        FBICode,Latitude,Longitude]
        print(test_features)
        x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,train_size=0.8,random_state=1)

            #Target Class to test
        knn_model = KNeighborsClassifier(n_neighbors=3)

        # Model Training
        knn_model.fit(x_train,y_train)
        float_list = [float(string_value) for string_value in test_features]

        # Prediction
        result = knn_model.predict([float_list])
        if result==1:
                msg = "There is no crime"
                return render(request, 'users/prediction.html', {'msg':msg})
        elif result==0:
                msg='There is crime'
                return render (request, 'users/prediction.html', {'msg':msg})
            
        else: 
            print("not valid")
            return render(request,'users/prediction.html',{})
    else:
        return render(request,'users/prediction.html',{})








