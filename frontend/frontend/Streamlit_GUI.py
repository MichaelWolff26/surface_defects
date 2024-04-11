import streamlit
from PIL import Image
import os
import requests


streamlit.header("Steel surface defect analyser")

streamlit.write(
    "This tool analyzes wether a steel surface has defects or not and uses Artificial Intelligence.  \nThe projectfiles can be viewed here:https://github.com/MichaelWolff26/surface_defects.git."
)

streamlit.write(
    "The Data is based on the Kolektor Surface-Defect Dataset 2 and is modified.  \nLicense: https://creativecommons.org/licenses/by-nc-sa/4.0/  \nThis is a non commercial website."
)


streamlit.header("1: Pick a sample image for analysis")

number = streamlit.slider("Move the slider to select a a sample image, the image is shown below")


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

data = streamlit.file_uploader("Or upload own surface image here and activate the toogle switch below, Size min. 633x228 pixel", type="png")

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
    streamlit.header("2: Click the Analyze button")
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
