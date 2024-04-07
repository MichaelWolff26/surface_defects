import torchvision
import os
import torch
import pandas as pd


def create_label_list(data_path):

    path_good = os.listdir(data_path + "\\good")
    df_good = pd.DataFrame({"file_name": [f"\\good\\{fn}" for fn in path_good]})
    df_good["label"] = 0
    path_defect = os.listdir(data_path + "\\defect")
    df_defect = pd.DataFrame({"file_name": [f"\\defect\\{fn}" for fn in path_defect]})
    df_defect["label"] = 1
    df_labels =  pd.concat([df_good, df_defect], axis=0)
    return df_labels


def read_data(
    data_path,
):  # Reading dataframe and image-data, return x_values, y_values as torch tensors
    df1 = create_label_list(data_path=data_path)
    df1=df1.reset_index()
    example_image_path = data_path + df1["file_name"][1]
    img_size = torchvision.io.read_image(example_image_path).size() # get image size
    x_values_size = (
        2 * len(df1)
    )  # x_values_size is 2 times the length of the original data, because of the flipping operation
    x_values = torch.zeros(
        x_values_size, img_size[0], img_size[1], img_size[2], dtype=torch.float32
    )  # creating x_values
    y_values = torch.zeros(x_values_size, dtype=torch.long)  # creating y_values

    c_var = 0

    for index, row in df1.iterrows():  # read the images to x_values, crop them to size, normalize the values (/255) and flip them; assign labels to y_values
        img_temp = torchvision.io.read_image(data_path + df1["file_name"][index])
        img_cropped = torchvision.transforms.functional.crop(
            img_temp, top=0, left=0, height=633, width=228
        )
        norm_img_temp = img_cropped / 255
        x_values[c_var] = norm_img_temp
        y_values[c_var] = df1["label"][index]
        x_values[c_var + 1] = torchvision.transforms.functional.hflip(norm_img_temp)
        y_values[c_var + 1] = df1["label"][index]
        c_var = c_var + 2
    return x_values, y_values, df1
