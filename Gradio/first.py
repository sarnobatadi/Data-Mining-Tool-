import gradio as gr

def greet(name):
    return "Hello " + name + "!"

def good(name):
    return "Hello " + name + "!"

demo = gr.Interface(fn=greet, inputs="text", outputs="text")
demo = gr.Interface(fn=good, inputs="text", outputs="text")

demo.launch()   