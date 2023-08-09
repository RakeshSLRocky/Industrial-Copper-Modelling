# import the dependencies
import streamlit as st
import pickle
import numpy as np
import pickle
from streamlit_option_menu import option_menu
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import warnings
warnings.filterwarnings('ignore')

#tittle
st.markdown("<h2 style= 'color: #F4511E;font-size: 48px;font-weight :900'><b>INDUSTRIAL COPPER MODELING</b></h2>",
                unsafe_allow_html=True)


#pickle file
with open(r"model1.pkl", 'rb') as file:
    loaded_model = pickle.load(file)
with open(r'scale.pkl', 'rb') as f:
    scaler_loaded = pickle.load(f)

with open(r"t.pkl", 'rb') as f:
    t_loaded = pickle.load(f)

with open(r"s.pkl", 'rb') as f:
    s_loaded = pickle.load(f)

with open(r"model2.pkl", 'rb') as file:
   cloaded_model = pickle.load(file)



#selected options
status_options = [1,0]
item_type_options = [0, 1, 2, 3, 4, 5, 6]
application_options = [10., 41., 28., 59., 15., 4., 38., 56., 42., 26., 27., 19., 20., 66., 29., 22., 40., 25., 67., 79., 3., 99., 2., 5., 39., 69., 70., 65., 58., 68.]
country_options = [28., 25., 30., 32., 38., 78., 27., 77., 113., 79., 26., 39., 40., 84., 80., 107., 89.]
product=[611112, 611728, 628112, 628117, 628377, 640400, 640405, 640665, 611993, 929423819, 1282007633, 1332077137, 164141591, 164336407, 
                     164337175, 1665572032, 1665572374, 1665584320, 1665584642, 1665584662, 
                     1668701376, 1668701698, 1668701718, 1668701725, 1670798778, 1671863738, 
                     1671876026, 1690738206, 1690738219, 1693867550, 1693867563, 1721130331, 1722207579]


selected = option_menu(
    menu_title=None,
    options=["PREDICT SELLING PRICE", "PREDICT STATUS"],
    icons=["currency-rupee", "chevron-contract"],
    default_index=0,
    orientation="horizontal"
)
#selling price
#tab1, tab2 = st.tabs(["PREDICT SELLING PRICE", "PREDICT STATUS"]) 
#with tab1:
if selected=="PREDICT SELLING PRICE":
    col1,col2,col3=st.columns([5,2,5])
    with col1:
        st.write(' ')
        quantity_tons = st.number_input(("Enter quality tons(Min:611728 & Max:1722207579)"))
        customer = st.number_input("Enter customer ID (Min:12458, Max:30408185)")
        status = st.selectbox("Select status(won:1 lost:0)", status_options,key=1)
        item_type = st.selectbox("Select item Type(W:0,WI:1,S:2,Others:3,PL:4,IPL:5,SLAWR:6)", item_type_options,key=2)
        application = st.selectbox(" Select Application",sorted(application_options),key=3)
    with col3:
        thickness = st.number_input("Enter thickness (Min:0.18 & Max:400)")
        width = st.number_input("Enter width (Min:1, Max:2990)")
        country = st.selectbox("Country", sorted(country_options),key=4)
        product_ref = st.selectbox("Product Reference", product,key=5)
    b=st.columns([3])
    Price = st.button("PREDICT SELLING PRICE")
    if Price:
        new_sample= np.array([[(float(quantity_tons)),np.log(float(customer)),float(status),float(item_type),np.log(float(application)),np.log(float(thickness)),np.sqrt(float(width)),int(country),int(product_ref)]])
        new_sample_ohe = t_loaded.transform(new_sample[:, [7]]).toarray()
        new_sample_be = s_loaded.transform(new_sample[:, [8]]).toarray()
        new_sample = np.concatenate((new_sample[:, [0,1,2, 3, 4, 5, 6,]], new_sample_ohe, new_sample_be), axis=1)
        new_sample1 = scaler_loaded.transform(new_sample)
        new_pred = loaded_model.predict(new_sample1)[0]
        st.write('## :red[Predicted selling price:] ', (new_pred))

    #with st.sidebar:
        #st.write('W':'0',WI1,S2,Others3,PL4,IPL5,SLAWR6)



#status
elif selected=="PREDICT STATUS":
    col1,col2,col3=st.columns([5,2,5])
    with col1:
        st.write(' ')
        quantity_tons = st.number_input(("Enter quality tons(Min:611728 & Max:1722207579)"),key=10)
        customer = st.number_input("Enter customer ID (Min:12458, Max:30408185)",key=11)
        item_type = st.selectbox("Select item Type(W:0,WI:1,S:2,Others:3,PL:4,IPL:5,SLAWR:6)", item_type_options,key=12)
        application = st.selectbox(" Select Application",sorted(application_options),key=13)
        price=st.number_input("Enter Selling Price")
    with col3:
        thickness = st.number_input(("Enter thickness (Min:0.18 & Max:400)"),key=15)
        width = st.number_input("Enter width (Min:1, Max:2990)",key=16)
        country = st.selectbox("Country", sorted(country_options),key=17)
        product_ref = st.selectbox("Product Reference", product,key=18)

    c=st.columns([3])
    status=st.button("PREDICT STATUS")
    if status:
        new_sample = np.array([[(float(quantity_tons)),(float(customer)), country,float(item_type),float(application),(float(thickness)),float(width),int(product_ref),int(price)]])
        predict=cloaded_model.predict(new_sample)[0]
        #st.write(predict)
        if predict==1:
            st.write('## :red[STATUS IS WON]')
        else:
            st.write('## :red[STATUS IS LOST]')  
