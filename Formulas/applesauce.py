




def transform_df(df): # this will create a binary encoding for fatalities in an accident,
    # 1 for a fatality was present
    # and 0 for no fatality present
    df['INJURIES_FATAL'] = df['INJURIES_FATAL'].apply(yes_no_converter)
    # df['y'] = df['y'].apply(yes_no_converter)
    return df

def score_report(ytrue, ypred):
    print("Accuracy Score: ", accuracy_score(ytrue, ypred))
    print("Precision Score: ", precision_score(ytrue, ypred)) # a little difficult for this to tell us much given the small ratio of fatalities to accidents
    print("Recall Score: ", recall_score(ytrue, ypred)) # recall score is helpful, we can see that we are modeling almost 63% of accidents where there is a fatality     
    print("F1 Score: ", f1_score(ytrue, ypred))
    pass

def getList(dict):
    return dict.keys()

def model_opt(models, x, y, xtest, ytest):
    for model in models:
        pipe = Pipeline(steps=[('model', model)])
        fit = pipe.fit(x, y)
        ypred = model.predict(xtest)
        print(model," ", fit.score(x, y))
        plot_confusion_matrix(model, xtest, ytest, values_format='1')
        plt.show()
        score_report(ytest, ypred)
    pass

def model_scoring(models, x_train_resampled, y_train_resampled, x_test, y_test):
    for model in models:
        train_score = model.score(x_train_resampled, y_train_resampled)
        test_score = model.score(x_test, y_test)
        avg_score = (train_score + test_score)/2
        print(model ,train_score, test_score, avg_score)

def cost_benefit_analysis(model, X_test, y_test):
    y_preds = model.predict(X_test)
    label_dict = {"TP":0, "FP": 0, "TN": 0, "FN": 0}
    for yt, yp in zip(y_test, y_preds):
        if yt==yp:
            if yt==1:
                label_dict["TP"] += 1
            else:
                label_dict["TN"] += 1
        else:
            if yp==1:
                label_dict["FP"] += 1
            else:
                label_dict["FN"] += 1
    cb_dict = {"TP": 50, "FP": -10, "TN": 0, "FN": -60}
    total = 0
    for key in label_dict.keys():
        total += cb_dict[key]*label_dict[key]
    return total / sum(label_dict.values())
