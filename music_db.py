import requests
from bs4 import BeautifulSoup
import re
import sqlite3
import urllib.request

# res = requests.get('https://www.azlyrics.com/lyrics/samsmith/dancingwithastranger.html')
#singer와 title을 적으면 가사를 찾아주는 함수
def find_music(singer, title):
    # singer = input('singer : ')
    # title = input('title : ')


    singer = singer.lower()
    singer = re.sub(r'\s+|[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]|the', '', singer)

    title = title.lower()
    title = re.sub(r'\s+|[-=+,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》]', '', title)

    print(singer + "  " + title)
    url = "https://www.azlyrics.com/lyrics/" + singer + "/"+title+".html"
    res = requests.get(url)
    print(url)

    soup = BeautifulSoup(res.content, 'html.parser')
    soup_text = str(soup)
    p = re.compile('that. -->(.*?)</div>', re.DOTALL)
    lyric = p.findall(soup_text)
    # print(lyric)

    lyric = str(lyric).split("<br/>")
    lyric_text = ""
    for i in range(len(lyric)):
        lyric[i] = re.sub(r"\\n|\n|\[|\]|\"|\\r|\r","",lyric[i])
        lyric_text = lyric_text +" "+lyric[i]
    return singer, title, lyric_text

# db에 data를 입력하고, 에러가 날 경우 이를 출력하는 함수
def saveDBtable(db, data):
    conn = sqlite3.connect(db)
    cur = conn.cursor()
    try:
        sql = "insert into song (singer,title,lyric) values (?,?,?)"
        cur.execute(sql, data)
    except sqlite3.Error as error:
        print(error)
    conn.commit()
    conn.close()

# music.db 에 song(singer text, title text, lyric text, primary key (singer, title))을 생성하는 함수
def createDBtable():
    conn = sqlite3.connect('music.db')
    try:
        cur = conn.cursor()
        cur.execute("create table song(singer text, title text, lyric text, PRIMARY KEY (singer, title))")
        print("create new DB named music.db")
        cur.close()
    except sqlite3.Error as error:
        print("DB named music.db is already exist")
    finally:
        if conn:
            conn.close()
            print("def createDBtable end success")

#빌보드를 가져와서 txt파일을 만든다.
def get_billboard_data():
    pass

if __name__ == "__main__":
    db = "music.db"
    createDBtable()

    f = open("billboard.txt", 'r')
    lines = f.readlines()
    billboard_list = []
    for line in lines:
        print(line)
        temp = re.sub(r'\n|\s+','',line)
        temp = temp.split('-')
        billboard_list.append(temp)
    f.close()

    print(billboard_list)
    for song_list in billboard_list :
        singer = song_list[0]
        title = song_list[1]
        singer, title, lyric_text = find_music(singer, title)
        if lyric_text != " ":
            print(lyric_text)
            data = (singer, title, lyric_text)
            saveDBtable(db, data)
        else:
            print("no lyric data")


