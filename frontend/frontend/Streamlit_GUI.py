import streamlit
from PIL import Image
import os
import requests


streamlit.header("Steel surface defect analyser")
streamlit.write(
    "The Data is based on the Kolektor Surface-Defect Dataset 2 and is modified"
)
streamlit.write("License: https://creativecommons.org/licenses/by-nc-sa/4.0/")
streamlit.write("This is a non commercial website")

number = streamlit.slider("Pick one of the 100 sample images")


data_path = f"{os.getcwd()}/Dataset-trafo"

list_namesg = []
list_namesd = []
list_samples = []
for f in os.listdir(data_path + "/good/"):
    list_namesg.append("/good/" + f)
for f in os.listdir(data_path + "/defect/"):
    list_namesd.append("/defect/" + f)
for i in range(50):
    list_samples.append(list_namesg[i])
    list_samples.append(list_namesd[i])

with Image.open(data_path + list_samples[number]) as img:
    img.load()
    data2=img

data = streamlit.file_uploader("Or upload own surface image", type="png")

own_data = streamlit.toggle("Use own surface image")

if (data or data2) is not None:
    if data is not None and own_data:
        with Image.open(data) as img:
            img.load()
            test=img
            file = data.getvalue()
            filename = data.name
    if data2 is not None and not own_data:
        test = data2
        filename = data_path + list_samples[number]
        with open(data_path + list_samples[number], "rb") as img:
            file=img.read()

    if not (own_data and not data):
        streamlit.image(test)

    if streamlit.button("Analyze"):
        
        url = "http://backend:8000/upload_image"
        payload = {"filename": filename}
        response = requests.post(url, params=payload, files={"img_file": file})
        class_var = response.content.decode()

        c1, c2 = streamlit.columns(2)
        c1.header("Surface Quality:")

        if class_var == "0":
            c2.image("Streamlit_data/IO.png", width=140)
        if class_var == "1":
            c2.image("Streamlit_data/NIO.png", width=140)
