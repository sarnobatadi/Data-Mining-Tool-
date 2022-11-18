# Data-Mining-Tool Desktop Application with EEL Python 

## Data Mining Tool 
1. Data Mining Tool Contains Implementation of following Data Mining Phases
  <br> a. Data Preprocessing 
  <br> b. Data Cleaning 
  <br> c. Data Integration 
  <br> d. Classification
  <br> e. Apriori Rule Generation
  <br> f. Web Mining 



## About EEL Python 
1.	Eel is a little Python library for making simple Electron-like offline HTML/JS GUI apps, with full access to Python capabilities and libraries.
2.	Eel hosts a local webserver, then lets you annotate functions in Python so that they can be called from Javascript, and vice versa.
3.	Eel is designed to take the hassle out of writing short and simple GUI applications
4.	Installation - 
5.	Install from pypi with pip:
pip install eel
6.	To include support for HTML templating, currently using Jinja2:
    <br> `pip install eel[jinja2]`
7.	Directory Structure <br>
  a.	Main Project File -/ main.py (for initial python code and defining python functions with @eel.expose header ) <br>
  b.	Main Project File -/ web / index.html (initial html files on start of application)  <br>
  c.	Main Project File -/ web / styles.css (css file)  <br>
  d.	Main Project File -/ web / script.js (js file for fetching data from user and for connection with python script )

8.	Stating Application 
  <br> a.	Run main.py to start desktop app 
  <br> ``` python main.py ``` <br>
  b.	This will open desktop window with content of index.html

Reference – 
EEL Python Docs - https://github.com/ChrisKnott/Eel  
