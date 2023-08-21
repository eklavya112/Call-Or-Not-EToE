# Model Deployment on Streamlit

import streamlit as st
from modules import *
df = get_data("Bank.csv",";")
final_df = ftr_eng_sel(df)

#Training the model
trained_model(final_df,[0,1,2,3,5,6])

model = joblib.load("trained_model.pkl")
J = ['housemaid', 'services', 'admin.', 'blue-collar', 'technician',
       'retired', 'management', 'unemployed', 'self-employed', 'unknown',
       'entrepreneur', 'student']

M = ['Second_Quarter', 'Third_Quarter', 'Fourth_Quarter',
       'First_Quarter']

D = ['no', 'unknown', 'yes']
MAR =['married', 'single', 'divorced', 'unknown']

ANY_L = ['yes','no','unknown']


EDU = ['elementary', 'high.school', 'professional.course', 'unknown',
       'university.degree', 'illiterate']


st.title("Make a Call or Not (New Bank Policy Promotion)")
keys = (e for e in range(1,100))
col1,col2 = st.columns(2,gap="large")
with col1:
    age = st.number_input('AGE',key=next(keys))

with col2:
    education = st.selectbox('EDUCATION',sorted(EDU),key=next(keys))

col3,col4 = st.columns(2,gap="large")
with col3:
    marital = st.selectbox('MARITAL_STATUS',sorted(MAR),key=next(keys))

with col4:
    job = st.selectbox('JOB',sorted(J),key=next(keys))

col5,col6,col7 = st.columns(3,gap="large")

with col5:
    any_loans = st.selectbox("ANY LOANS",sorted(ANY_L),key=next(keys))
with col6:
    month = st.selectbox("TIME PERIOD",sorted(M),key=next(keys))
with col7:
    defaulter = st.selectbox("DEFAULTER",sorted(D),key=next(keys))

if st.button('Prediction',key=next(keys)):
            df = pandas.DataFrame(
                {'job': [job], 'marital': [marital],'education':[education],'default':[defaulter],
                 'log_age': [numpy.log(age)],'Any_Loans': [any_loans], 'Month':
                   [month]})
            result = model.predict_proba(df)
            yes = (result[0][1])*100
            no = (result[0][0])*100
            st.title("Success Chances : ")
            st.title("YES: "+ str(round(yes))+" %")
            st.title("NO: " + str(round(no))+ " %")
            

    

    

