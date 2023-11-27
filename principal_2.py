import pandas as pd
#import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
df = pd.read_csv("Salary_Data.csv")
st.title("Welcome to Dashboard of Salary Data Prediction :sunglasses:")
st.write("by: H.Marques")

with st.container():
    st.header("Import the packages:")
    code = '''
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    import sklearn.metrics as metrics

            '''
    st.code(code, language='python')

with st.container():
    df = pd.read_csv("Salary_Data.csv")
    df
    st.header("Reading the CSV:")
    code = '''
    df = pd.read_csv("Salary_Data.csv")
            '''
    st.code(code, language='python')

with st.container():
    st.header("Droping the null values, because they're full nullable rows")

    code = '''
    df[df["Gender"].isnull()]

    df.dropna(inplace = True)

    df.reset_index(drop=True,inplace=True)
            '''
    st.code(code, language='python')

    df[df["Gender"].isnull()]

    df.dropna(inplace = True)

    df.reset_index(drop=True,inplace=True)

with st.container():

    st.header("Transformating Age,Years of XP and Salary columns to int64 format")

    code='''
    df["Age"] = df["Age"].astype(np.int64)
    df["Years of Experience"] = df["Years of Experience"].astype(np.int64)
    df["Salary"] = df["Salary"].astype(np.int64)
        '''
    st.code(code, language='python')

    df["Age"] = df["Age"].astype(np.int64)
    df["Years of Experience"] = df["Years of Experience"].astype(np.int64)
    df["Salary"] = df["Salary"].astype(np.int64)

with st.container():

    st.header("Replacing integer values for gender")
    st.markdown("1-Male, 2-Female")
    code='''
    df["Gender"].replace({"Male":1,"Female":2},inplace=True)
        '''
    st.code(code, language='python')
    df["Gender"].replace({"Male":1,"Female":2},inplace=True)


with st.container():
    st.header("Replacing integer values for Education level:")
    st.markdown("1-Bachelor's, 2-Master's, 3-PhD")
    code='''
    df["Education Level"].replace({"Bachelor's":1,"Master's":2,"PhD":3},inplace=True)
        '''
    st.code(code, language='python')
    df["Education Level"].replace({"Bachelor's":1,"Master's":2,"PhD":3},inplace=True)

with st.container():
    st.header("Seeing dataframe:")
    code='''
    df
        '''
    st.code(code, language='python')
    df

with st.container():
    st.markdown("We've a lot of same titles for diferent works like Director of Engineering and Director oh HR.")
    st.markdown("Both are Directors, so we'll try to find a correlation between these titles and the salaries")
    st.markdown("The same logic will be applied to the seniority, because we've a lot of juniors and seniors in the dataset")
    code='''
    df["Job Title"].unique()

    df[df["Job Title"] == "Director of Engineering"]
        '''
    st.code(code, language='python')

    df["Job Title"].unique()

    df[df["Job Title"] == "Director of Engineering"]

with st.container():
    st.header("Creating a function that takes the Job Title column and returns the Junior or Senior seniority")
    st.caption("There are some Job title that don't have that information, so for that will be nullable values")
    code='''
    def seniority(title):
        size = len(title.split())
        if title.split()[0] == "Junior":
            return "Junior"
        elif title.split()[0] == "Senior":
            return "Senior"
        '''
    st.code(code, language='python')

    def seniority(title):
        size = len(title.split())
        if title.split()[0] == "Junior":
            return "Junior"
        elif title.split()[0] == "Senior":
            return "Senior"

with st.container():
    st.header("Applying the seniority funcion to Job Titles and creating a column with that information")
    code='''
    df["Seniority"] = df["Job Title"].apply(seniority)
         '''
    st.code(code, language='python')

    df["Seniority"] = df["Job Title"].apply(seniority)

with st.container():
    st.header("Now we'll solve the nullable values in the seniority column newly created")
    st.markdown("For most companies, who got 5 or less years of experience is junior, between 5 and 10 is medior, and 10+ is senior ")
    code='''
    for row in range(len(df["Seniority"])):
        if df["Seniority"][row] == None:
            if df["Years of Experience"][row] <= 5:
                df["Seniority"][row] = "Junior"
            elif df["Years of Experience"][row] > 5 and df["Years of Experience"][row] <= 10:
                df["Seniority"][row] = "Medior"
            else: 
                df["Seniority"][row] = "Senior"
    '''
    st.code(code, language='python')

    for row in range(len(df["Seniority"])):
        if df["Seniority"][row] == None:
            if df["Years of Experience"][row] <= 5:
                df["Seniority"][row] = "Junior"
            elif df["Years of Experience"][row] > 5 and df["Years of Experience"][row] <= 10:
                df["Seniority"][row] = "Medior"
            else: 
                df["Seniority"][row] = "Senior"

with st.container():
    st.header("Creating a function that takes the job title and returns the position in company")
    st.markdown("Ex: Director or manager")
    code='''
    def position(entry):
        size = len(entry.split())
        try:
            if entry.split()[1] != "of":
                return entry.split()[size-1]
            else:
                return entry.split()[0]
        except: 
            return entry
    '''
    st.code(code, language='python')

    def position(entry):
        size = len(entry.split())
        try:
            if entry.split()[1] != "of":
                return entry.split()[size-1]
            else:
                return entry.split()[0]
        except: 
            return entry
with st.container():
    st.header("Seeing dataframe:")
    code='''
    df
        '''
    st.code(code, language='python')
    df
with st.container():
    st.header("Applying the newly created funcion:")
    code='''
    df["Title"] = df["Job Title"].apply(position)
    '''
    st.code(code, language='python')

    df["Title"] = df["Job Title"].apply(position)

with st.container():
    st.markdown("Now, we'll group the mean values for each title for sort that values and replace them by integer values: ")
    code='''
    col = []
    means = []
    for i in range (len(df["Title"].unique())):
        means.append(df[df["Title"] == df["Title"][i]]["Salary"].mean())
        col.append(df["Title"].unique()[i])
    '''
    st.code(code, language='python')

    col = []
    means = []
    for i in range (len(df["Title"].unique())):
        means.append(df[df["Title"] == df["Title"][i]]["Salary"].mean())
        col.append(df["Title"].unique()[i])

with st.container():
    code='''
    jobs = {
        "Job":col,
        "Mean":means,
    }
    df_jobs = pd.DataFrame(jobs)
    '''
    st.code(code, language='python')

    jobs = {
        "Job":col,
        "Mean":means,
    }
    df_jobs = pd.DataFrame(jobs)

with st.container():
    code='''
    df_jobs = df_jobs.sort_values(by="Mean").reset_index(drop=True)
    df_jobs
    '''
    st.code(code, language='python')
    df_jobs = df_jobs.sort_values(by="Mean").reset_index(drop=True)
    df_jobs
    
with st.container():
    st.header("Seeing dataframe:")
    code='''
    df
        '''
    st.code(code, language='python')
    df

with st.container():

    st.header("Replacing title names by integer values:")
    code='''
        for i in range (len(df["Title"])):
            for t in range (len(df_jobs)):
                if df["Title"][i] == df_jobs["Job"][t]:
                    df["Title"][i] = t
        '''
    st.code(code, language='python')

    for i in range (len(df["Title"])):
        for t in range (len(df_jobs)):
            if df["Title"][i] == df_jobs["Job"][t]:
                df["Title"][i] = t
with st.container():
    st.header("Seeing dataframe:")
    code='''
    df
        '''
    st.code(code, language='python')
    df
with st.container():

    st.header("Making a astype in Title column to guarantee that values are integers")
    code='''
    df["Title"] = df["Title"].astype(np.int64)
    '''
    st.code(code, language='python')

    df["Title"] = df["Title"].astype(np.int64)

with st.container():

    st.header("Replacing the seniority titles to integers too ")
    code='''
    df["Seniority"].replace({"Senior":3,"Medior":2,"Junior":1},inplace=True)
    '''
    st.code(code, language='python')

    df["Seniority"].replace({"Senior":3,"Medior":2,"Junior":1},inplace=True)

with st.container():
    st.header("Seeing dataframe:")
    code='''
    df
        '''
    st.code(code, language='python')
    df

with st.container():

    st.header("Making a correlation between the values")
    code='''
    corr = df.corr(method="pearson",numeric_only=True)
    mask = np.triu(corr)
    fig, ax = plt.subplots()
    sns.heatmap(data=corr,annot=True,mask=mask,ax=ax,cmap="crest")
    ax.set_title("Correlations of variables")
    '''
    st.code(code, language='python')

    corr = df.corr(method="pearson",numeric_only=True)
    mask = np.triu(corr)
    fig, ax = plt.subplots()
    sns.heatmap(data=corr,annot=True,mask=mask,ax=ax,cmap="crest")
    ax.set_title("Correlations of variables")
    st.pyplot(fig)

with st.container():

    st.header("Boxplots for see outliers")
    code='''
    fig, ax = plt.subplots()
    sns.boxplot(data=corr,ax=ax)
    ax.set_xticks(np.arange(len(df.columns)),df.columns,rotation=90)
    ax.set_title("Boxplot of correlations")
    plt.tight_layout()
    plt.show()
    '''
    st.code(code, language='python')

    fig, ax = plt.subplots()
    sns.boxplot(data=corr,ax=ax)
    ax.set_xticks(np.arange(len(df.columns)),df.columns,rotation=90)
    ax.set_title("Boxplot of correlations")
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)

with st.container():

    st.header("Making a prediction of the variables for salaries")
    code='''
    label1 = (column) # <- Is just put the name of correlated column in that variable for make the prediction 
    label2 = "Salary"
    col1 = df[label1]
    col2 = df[label2]
    X = col1
    y = col2
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=46)
    X_train = np.array(X_train).reshape(-1,1)
    X_test = np.array(X_test).reshape(-1,1)
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    c = lr.intercept_
    m = lr.coef_
    Y_pred_train = m*X_train+c
    y_pred_train1 = lr.predict(X_train)
    fig, ax = plt.subplots(nrows=2)
    ax[0].scatter(x=col1,y=col2)
    ax[0].set_title(f"{label1} x {label2}")
    ax[0].set_ylabel(f"{label2}")
    ax[0].set_xlabel(f"{label1}")
    ax[1].scatter(x=X_train,y=y_train)
    ax[1].plot(X_train,y_pred_train1,c="r")
    ax[1].set_title(f"{label1} x {label2} w/Prediction")
    ax[1].set_ylabel(f"{label2}")
    ax[1].set_xlabel(f"{label1}")
    mae = metrics.mean_absolute_error(y_train,y_pred_train1)
    mse = metrics.mean_squared_error(y_train,y_pred_train1)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_train,y_pred_train1)
    print(f"For {label1} x {label2}:\n")
    print(f"Mae: {mae}\nMse: {mse}\nRmse: {rmse}\nR2: {r2}")

    plt.tight_layout()
    plt.show()
        
        
        '''
    st.code(code, language='python')

with st.container():
    opt = st.selectbox("Choose the variable to do a prediction:",("Age","Gender","Education Level","Years of Experience","Seniority","Title"))
    st.write("You selected:", opt)
    label1 = opt # <- Is just put the name of correlated column in that variable for make the prediction 
    label2 = "Salary"
    col1 = df[label1]
    col2 = df[label2]
    X = col1
    y = col2
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.4,random_state=46)
    X_train = np.array(X_train).reshape(-1,1)
    X_test = np.array(X_test).reshape(-1,1)
    lr = LinearRegression()
    lr.fit(X_train,y_train)
    c = lr.intercept_
    m = lr.coef_
    Y_pred_train = m*X_train+c
    y_pred_train1 = lr.predict(X_train)
    fig, ax = plt.subplots(nrows=2)
    ax[0].scatter(x=col1,y=col2)
    ax[0].set_title(f"{label1} x {label2}")
    ax[0].set_ylabel(f"{label2}")
    ax[0].set_xlabel(f"{label1}")
    ax[1].scatter(x=X_train,y=y_train)
    ax[1].plot(X_train,y_pred_train1,c="r")
    ax[1].set_title(f"{label1} x {label2} w/Prediction")
    ax[1].set_ylabel(f"{label2}")
    ax[1].set_xlabel(f"{label1}")
    mae = metrics.mean_absolute_error(y_train,y_pred_train1)
    mse = metrics.mean_squared_error(y_train,y_pred_train1)
    rmse = np.sqrt(mse)
    r2 = metrics.r2_score(y_train,y_pred_train1)
    st.write(f"For {label1} x {label2}:\n")
    st.write(f"Mae: {mae}")
    st.write(f"Mse: {mse}")
    st.write(f"Rmse: {rmse}")
    st.write(f"R2: {r2}")
    plt.tight_layout()
    plt.show()
    st.pyplot(fig)
    
        


