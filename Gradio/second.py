import gradio as gr
import pandas as pd

def predict(file_obj,Question):
    df = pd.read_csv(file_obj.name,dtype=str)
    return df

def main():
    io = gr.Interface(predict, ["file",gr.inputs.Textbox(placeholder="Enter Question here...")], "dataframe")
    io.launch()

if __name__ == "__main__":
    main()