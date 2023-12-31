import streamlit as st
import pandas as pd
import numpy as np

st.markdown("---")



html_temp_2 = """
<div style ="margin-top:20px"> <img src="https://miro.medium.com/v2/resize:fit:1000/1*GDjVt1eUGYVOxn1d04g7uw.jpeg" alt ="Car Image" style = "display:block;margin:auto; width:500px;height:auto;"> </div>
"""
st.markdown(html_temp_2,unsafe_allow_html=True)



st.markdown("---")

# title of the sidebar
html_temp = """
<div style="background-color:green;padding:10px">
<h2 style="color:white;text-align:center;">Car Price Prediction </h2>
</div>"""

st.sidebar.markdown(html_temp,unsafe_allow_html=True)


selected_algorithm = st.sidebar.selectbox("Select Algorithm", ["Random Forest", "XGBoost"], index = 0)

# Load the appropriate CSV file for the selected algorithm
if selected_algorithm == "Random Forest":
    data_filename = "rf_data.csv"
elif selected_algorithm == "XGBoost":
    data_filename = "xgb_data.csv"
else:
    st.error("Invalid Selection!")

df = pd.read_csv(data_filename)
st.header("Training Dataframe is below:")
st.markdown("---")
st.write(df.sample(5))


make_model = st.sidebar.selectbox("Select the Auto Brand - Model", df["make_model"].unique(), index = 28)
gearbox = st.sidebar.selectbox("Select the Gearbox", df["gearbox"].unique(), index = 1)
drivetrain = st.sidebar.selectbox("Select the Drivetrain", df["drivetrain"].unique(), index = 1)
power_kw = st.sidebar.number_input("Enter the power (in kW)", min_value = df["power_kW"].min(), max_value = df["power_kW"].max(), value = df["power_kW"].mode().iloc[0])
age = st.sidebar.number_input("Enter the Age", min_value = df["age"].min(), max_value = df["age"].max(), value = df["age"].mode().iloc[0])
empty_weight = st.sidebar.number_input("Enter the Empty Weight", min_value = df["empty_weight"].min(), max_value = df["empty_weight"].max(), value = df["empty_weight"].mode().iloc[0])
mileage = st.sidebar.number_input("Enter the Mileage", min_value = df["mileage"].min(), max_value = df["mileage"].max(), value = df["mileage"].mode().iloc[0])
gears = st.sidebar.number_input("Enter the Gears", min_value = df["gears"].min(), max_value = df["gears"].max(), value = df["gears"].mode().iloc[0])
cons_avg = st.sidebar.number_input("Enter the Average Consumption", min_value = df["cons_avg"].min(), max_value = df["cons_avg"].max(), value = df["cons_avg"].mode().iloc[0])
co_emissions = st.sidebar.number_input("Enter the Average CO Emissions", min_value = df["co_emissions"].min(), max_value = df["co_emissions"].max(), value = df["co_emissions"].mode().iloc[0])

# To load machine learning model
import pickle
from sklearn.ensemble import RandomForestRegressor

file_name = "rf_pipe_model"
# model_rf = pickle.load(open(file_name, "rb"))

model_xgb = pickle.load(open("xgb_pipe_model", "rb"))

my_dict = {"power_kW":power_kw,
           "age":age,
           "empty_weight": empty_weight,
           "mileage": mileage,
           "gears": gears,
           "cons_avg": cons_avg,
           "co_emissions": co_emissions,
           "make_model": make_model,
           "gearbox": gearbox,
           "drivetrain":drivetrain}

st.header("The values you selected is below")
st.markdown("---")
df = pd.DataFrame.from_dict([my_dict])
st.table(df)

import os
from PIL import Image

# Streamlit uygulaması
st.title("Make Model ve Resim Gösterme")

# make_model seçim kutusu
make_model = st.sidebar.selectbox("Select the Auto Brand - Model", df["make_model"].unique(), index=0)

# Seçilen make_model'den ilk kelimeyi al ve küçük harfe çevir
make_model_lower = make_model.split()[0].lower()

# Resimleri içeren klasör yolu
pictures_folder = "Picture"

# make_model'e ait resmi bul
png_image_path = os.path.join(pictures_folder, f"{make_model_lower}.png")
jpg_image_path = os.path.join(pictures_folder, f"{make_model_lower}.jpg")

# Resmi görüntüle
try:
    # Önce PNG resmi dene
    image = Image.open(png_image_path)
except FileNotFoundError:
    try:
        # PNG bulunamazsa JPG resmi dene
        image = Image.open(jpg_image_path)
    except FileNotFoundError:
        st.warning(f"Resim bulunamadı: {png_image_path} veya {jpg_image_path}")
        st.stop()

# Resmi 400x200 piksel boyutunda göster
image = image.resize((200, 100))

st.image(image, caption=make_model, use_column_width=True)