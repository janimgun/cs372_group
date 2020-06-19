""" 주식종목 뉴스(네이버 파이넌스) Crawling 하기 """

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os

os.chdir('C:\cs372_group')  # 디렉토리 주소

df_news = pd.DataFrame()


def crawler(company_code, date_max):
    global df_news
    page = 1
    print(company_code)

    while 1:

        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page)
        print(url)
        source_code = requests.get(url).text
        html = BeautifulSoup(source_code, "lxml")

        # 뉴스 제목
        titles = html.select('.title')
        title_result = []
        for title in titles:
            title = title.get_text()
            title = re.sub('\n', '', title)
            title_result.append(title)
        # print(title_result)
        # 뉴스 링크
        links = html.select('.title')

        link_result = []
        for link in links:
            add = 'https://finance.naver.com' + link.find('a')['href']
            link_result.append(add)

        # 뉴스 날짜
        dates = html.select('.date')
        date_result = [date.get_text() for date in dates]

        # 뉴스 매체
        sources = html.select('.info')
        source_result = [source.get_text() for source in sources]

        # 변수들 합쳐서 해당 디렉토리에 csv파일로 저장하기

        text_result = []
        for url_link in link_result:
            source_code2 = requests.get(url_link).text
            html2 = BeautifulSoup(source_code2, 'lxml')
            texts = html2.select('.scr01')
            tt = [text.get_text() for text in texts]
            ttt = re.sub('\n|\t|\'', '', tt[0])
            ttt = re.split('ⓒ|▶', ttt)
            # print(ttt)
            text_result.append(ttt[0])

        company_result = [str(company_code)] * len(date_result)

        result = {"코드": company_result, "날짜": date_result, "언론사": source_result, "기사제목": title_result,
                  "링크": link_result, "기사내용": text_result}
        df_result = pd.DataFrame(result)
        print(df_result)

        date_cut = -1
        # print(df_result.loc[len(df_result.index)-1,"날짜"])
        for i in range(len(df_result.index)):
            date_day = df_result.loc[i, "날짜"].lstrip().split(' ')[0]
            # date_time = df_result.loc[i,"날짜"].lstrip().split(' ')[1]
            date_day_y = date_day.split('.')[0]
            date_max_day_y = date_max.split('.')[0]
            date_day_m = date_day.split('.')[1]
            date_max_day_m = date_max.split('.')[1]
            date_day_d = date_day.split('.')[2]
            date_max_day_d = date_max.split('.')[2]

            if (date_day_y < date_max_day_y) or (date_day_m < date_max_day_m) or (date_day_d < date_max_day_d):
                date_cut = i
                break
        if date_cut != -1:
            df_result = df_result.iloc[:date_cut]
            df_news = pd.concat([df_news, df_result])
            break
        page += 1
        df_news = pd.concat([df_news, df_result])

    df_news.to_csv('news_crawling.csv', mode='w', encoding='utf-8-sig')
    # 종목 리스트 파일 열기


# 회사명을 종목코드로 변환

def convert_to_code(company, maxpage):
    data = pd.read_csv('company_list.txt', dtype=str, sep='\t')  # 종목코드 추출
    company_name = data['회사명']
    keys = [i for i in company_name]  # 데이터프레임에서 리스트로 바꾸기

    company_code = data['종목코드']
    values = [j for j in company_code]

    dict_result = dict(zip(keys, values))  # 딕셔너리 형태로 회사이름과 종목코드 묶기

    pattern = '[0-9a-zA-Z가-힣]+'
    # if bool(re.match(pattern, company)) == True:  # Input에 이름으로 넣었을 때
    if bool(str(company) in keys) == True:
        company_code = dict_result.get(str(company))
        crawler(company_code, maxpage)


    else:  # Input에 종목코드로 넣었을 때
        company_code = str(company)
        crawler(company_code, maxpage)


def main():
    # info_main = input("=" * 50 + "\n" + "실시간 뉴스기사 다운받기." + "\n" + " 시작하시려면 Enter를 눌러주세요." + "\n" + "=" * 50)

    company = "삼성전자"
    max_date = "2020.06.18"

    convert_to_code(company, max_date)


main()
