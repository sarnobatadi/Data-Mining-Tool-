import gradio as gr
import pandas as pd


def predict(file_obj,Question,Select_Columns):
    df = pd.read_csv(file_obj.name,dtype=str)
    list(df.columns)
    return df,0

demo = gr.Interface(predict,["file","number",gr.inputs.Dropdown(["gpt2", "gpt-j-6B"])],["dataframe","number"])
demo.launch()