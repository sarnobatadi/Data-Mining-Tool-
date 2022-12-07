import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse, urljoin

def is_valid(url):
    parsed = urlparse(url)
    return bool(parsed.netloc) and bool(parsed.scheme)

urls = []


def dfscrawl(url):
    res = ""
    #Depth First Search
    url = ''
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    for link in soup.find_all('a'):
        weblink = link.get('href')
        if(is_valid(weblink)):
            # print("\nNode  : ",str(weblink))
            res +="\nNode  : " + str(weblink)
            reqs = requests.get(weblink)
            nextsoup = BeautifulSoup(reqs.text, 'html.parser')
            print("\nExploring Node  : ")
            res +="\nExploring Node  : "
            for link in nextsoup.find_all('a'):
                weblink = link.get('href')
                if(is_valid(weblink)):
                    # print(weblink)
                    res += "\n" + str(weblink)
    print(res)
    return res

def bfscrawl(url):              
    # url = 'https://softa.org.in/'
    res = ""
    reqs = requests.get(url)
    soup = BeautifulSoup(reqs.text, 'html.parser')
    #Breadth First Search
    print("All Weblinks of Seed Link : ")
    res += "All Weblinks of Seed Link : "
    for link in soup.find_all('a'):
        weblink = link.get('href')
        
        if(is_valid(weblink)): 
            print(str(weblink))
            res += "\n" + str(weblink)
            urls.append(weblink)

    for link in urls:
        reqs = requests.get(link)
        soup = BeautifulSoup(reqs.text, 'html.parser')
        print("\nAll Weblinks of: ",str(link))
        res += "All Weblinks of Seed Link : " + str(link)
        for l in soup.find_all('a'):
            weblink = l.get('href')
            
            if(is_valid(weblink)):
                res += "\n" + str(weblink)
                # print(weblink)
    print(res)
    return res
                
                
            
            

        
# dfscrawl('https://softa.org.in/')
# bfscrawl('https://softa.org.in/')