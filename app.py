import streamlit as st
from tensorflow.keras.models import load_model
import dill
import pandas as pd

model = load_model('notebook/ann_class_model.keras')

with open('notebook/transformer.pkl','rb') as f:
    transformer=dill.load(f)


st.title('Bank customer churn predictor')
Cred_score =st.number_input('CreditScore') 
Country =st.selectbox('Country',['France','Spain','Germany']) 
Gender =st.selectbox('Gender',['Male','Female']) 
Age =st.slider('Age',18,100,25) 
Tenure =st.number_input('Tenure',0,10,3) 
Balance =st.number_input('Balance') 
NumOfProducts =st.slider('NumOfProducts',0,4) 
HasCC =st.selectbox('HasCreditCard',[0,1]) 
IsActive =st.selectbox('IsActiveMember',[0,1]) 
EstimatedSalary =st.number_input('EstimatedSalary') 

input_data={
    'CreditScore' : Cred_score, 
    'Geography' : Country,       
    'Gender' : Gender,          
    'Age'  : Age,           
    'Tenure' : Tenure,         
    'Balance' : Balance,         
    'NumOfProducts' : NumOfProducts,     
    'HasCrCard' : HasCC,      
    'IsActiveMember' : IsActive,   
    'EstimatedSalary' : EstimatedSalary
}

new_data_df = pd.DataFrame([input_data],columns=input_data.keys())

new_data_t = transformer.transform(new_data_df)

pred = model.predict(new_data_t)

if st.button('Predict'):
    if pred < 0.5:
        st.success("Customer will not churn")
    else:
        st.error('Customer likely to churn')