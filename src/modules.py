from packages import *
def get_data(filepath:str,delimeter_type=None)->pandas.core.frame.DataFrame:
    '''
    This Function will return the data in DataFrame format.
    '''
    df = pandas.read_csv(filepath,delimiter=delimeter_type)
    return df


def get_details(df:pandas.core.frame.DataFrame) :
    '''
    This Function will provide a txt file containing details about the dataset like no of columns, dataset shpae, datatypes 
    and summary.
    '''
    file = open("DataSet Details.txt","w")
    file.write("Shape of Dataset: "+str(df.shape)+"\n")
    file.write("\n")
    file.write("Columns: "+str(list(df.columns))+"\n")
    file.write("\n")
    file.write("Data type of Columns: \n"+str(df.dtypes)+"\n")
    file.write("\n")
    file.write("No. of missing values: \n"+str(df.isnull().sum())+"\n")
    file.write("\n")
    file.write("---------Summary of Continious Variables----------\n")
    file.write("\n")
    file.write(str(df.describe())+"\n")
    file.write("\n")
    file.write("---------Summary of Categorical Variables----------\n")
    file.write("\n")
    for col in df:
        if df[col].dtype=='object':
            file.write(col+":\n")
            file.write(str(df[col].value_counts())+"\n")
            file.write("\n")
        else:
            pass
            
    file.close()      
    return 


def get_plot(df :pandas.core.frame.DataFrame, variable_list:list):
    '''
    This function will create and save plots for given variables in the list.
    Boxplot and Histogram -> Continious Variable
    Bar Plot -> Categorical Variable
    '''
    for col in variable_list:
        if df[col].dtype =='object':
                labels = list(df[col].unique())
                count = list(df[col].value_counts())
                seaborn.barplot(df,y=labels,x=count,orient='h')
                matplotlib.pyplot.title(col)
                matplotlib.pyplot.savefig(str(col)+"_barplot")
                matplotlib.pyplot.show()
        if df[col].dtype == ("int64" or "float64"):
                seaborn.boxplot(df[col])
                matplotlib.pyplot.title(col)
                matplotlib.pyplot.savefig(str(col)+"_boxplot")
                matplotlib.pyplot.show()
                seaborn.histplot(df[col])
                matplotlib.pyplot.savefig(str(col)+"_histplot")
                matplotlib.pyplot.show()
    return




def ftr_eng_sel(df:pandas.core.frame.DataFrame)->pandas.core.frame.DataFrame:
    '''
    Only important features were selected after doing variable analysis. One of the key aspects of model building.
    It will also return a txt file containing key aspects of feature engineering.
    '''
    #log transformation for age gave good results and looked like normal distribution
    df['log_age'] = numpy.log(df['age'])
    df["Any_Loans"] = df["housing"]+df["loan"]
    df["Any_Loans"].replace(["yesno","noyes","yesyes"],"yes",inplace=True)
    df["Any_Loans"].replace(["nono",],"no",inplace=True)
    df["Any_Loans"].replace(["nounknown",],"unknown",inplace=True)
    df["Any_Loans"].replace(["unknownno",],"unknown",inplace=True)
    df["Any_Loans"].replace(["unknownyes",],"yes",inplace=True)
    df["Any_Loans"].replace(["yesunknown",],"yes",inplace=True)
    df["Any_Loans"].replace(["unknownunknown",],"unknown",inplace=True)
    #devided all months in four quarters
    df['Month'] = df['month']
    df['Month'] = df['Month'].replace(['jan','feb','mar'],'First_Quarter')
    df['Month'] = df['Month'].replace(['apr','may','jun'],'Second_Quarter')
    df['Month'] = df['Month'].replace(['jul','aug','sep'],'Third_Quarter')
    df['Month'] = df['Month'].replace(['oct','nov','dec'],'Fourth_Quarter')
    df.drop('month',axis=1,inplace=True)
    
    # I am merging some classes together
    df['education'] = df['education'].replace(['basic.4y','basic.6y','basic.9y'],'elementary')
    # 1 for yes and 0 for no
    df['y'] = df['y'].replace(['no'],0)
    df['y'] = df['y'].replace(['yes'],1)
    
    for i in range(len(df['campaign'])):
        if (df['campaign'][i]>=1 and df['campaign'][i]<=5):
            df['campaign'][i]="Significance"

        elif (df['campaign'][i]>5 and df['campaign'][i]<=10):
            df['campaign'][i]="Frequent"

        else:
            df['campaign'][i]="Too Frequent"
    
    df.rename(columns={'campaign':"FrequencyOfPrvCalls"},inplace=True)
    # file = open("Feature Engineering Details.txt","w")
    # file.write("1. Log transformation is used on age column."+"\n")
    # file.write("2. housing and loan column is merged and made a new column called any_loans."+"\n")
    # file.write("3. All the contact months are devided in four quarters."+"\n")
    # file.write("4. No of camoaign calls are devided in three categories, significant (1-5), frequent (5-10) and too frequent (10+)"+"\n")
    # file.write("5. Following is a list of all final selected columns for training the model"+"\n")
    # file.write("[Job,marital,education,default,month,age(log),y(target),any_loans(housing+personal),Frequencyofprvcalls(campaign calls)]")
    # file.close()
    return df[['job','marital','education','default','y','log_age','Any_Loans','Month']]




def train_test(df:pandas.core.frame.DataFrame)->pandas.core.frame.DataFrame:
    '''
    This function takes dataframe as input and returns four dataframes: train_x, test_x, train_y,test_y
    '''
    x = df.drop('y',axis=1)
    y = df['y']
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.15,random_state=11,stratify=y)
    return x_train,x_test,y_train,y_test

# Model Building and training
import joblib
def trained_model(df:pandas.core.frame.DataFrame,element_indexes):
    '''
    This function will return the complete model with pipeline used and element _index are the index of column which shuould be encoded.
    '''
    x = df.drop('y',axis=1)
    y = df['y']
    step1 = ColumnTransformer([
            ('ohe',OneHotEncoder(sparse=False,handle_unknown='ignore'),element_indexes)
          ],remainder='passthrough')
        
    step2 = GaussianNB()
        
    pipe = Pipeline([
        ('step1',step1),
        ('step2',step2)
    ])
    pipe.fit(x,y)
    joblib.dump(pipe,"trained_model.pkl")
    return


