import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

urls = []


def dfscrawl():
    #Depth First Search
    url = 'https://softa.org.in/'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    for link in soup.find_all('a'):
        weblink = link.get('href')
        if(is_valid(weblink)):
            print("\nNode  : ",str(weblink))
            reqs = requests.get(weblink)
            nextsoup = BeautifulSoup(reqs.text, 'html.parser')
            print("\nExploring Node  : ")
            
            for link in nextsoup.find_all('a'):
                weblink = link.get('href')
                if(is_valid(weblink)):
                    print(weblink)


def bfscrawl():              
    url = 'https://softa.org.in/'
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    #Breadth First Search
    print("All Weblinks of Seed Link : ")
    for link in soup.find_all('a'):
        weblink = link.get('href')
        
        if(is_valid(weblink)): 
            print(str(weblink))
            urls.append(weblink)

    for link in urls:
        reqs = requests.get(link)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        print("\nAll Weblinks of: ",str(link))
        for l in soup.find_all('a'):
            weblink = l.get('href')
            
            if(is_valid(weblink)):
                print(weblink)
                
                
            
            

        
