""" 주식종목 뉴스(네이버 파이넌스) Crawling 하기 """

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os
import datetime

os.chdir('C:\cs372_group')  # 디렉토리 주소

df_news = pd.DataFrame()

def crawler(company_code, date_max):
    global df_news
    page = 1

    while 1:
        df_result = pd.DataFrame()
        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page)
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
        date_day = []
        date_time = []
        for i in date_result:
            date_day.append(i.lstrip().split(' ')[0])
            date_time.append(int(i.lstrip().split(' ')[1].split(':')[0]))

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

        result = {"코드": company_result, "날짜": date_day,"시간": date_time, "언론사": source_result, "기사제목": title_result,
                  "링크": link_result, "기사내용": text_result}
        df_result = pd.DataFrame(result)
        df_result = df_result.set_index("코드")

        date_datetime = datetime.datetime.strptime(date_max, '%Y.%m.%d') # max datetime
        date_before = date_datetime + datetime.timedelta(days=-1)
        date_before = date_before.strftime("%Y.%m.%d")
        pick_index = []
        for i in range(len(df_result.index)):
            if datetime.datetime.strptime(df_result.iloc[i]["날짜"], '%Y.%m.%d') > date_datetime:
                pick_index.append(i)
        df_result = df_result.iloc[pick_index,:]
        print(df_result)
        if df_result.empty:
            break
        else:
            link_result = []
            for i in range(len(df_result.index)):
                link_result.append(df_result.iloc[i]["링크"])
                # 뉴스 링크에 있는 기사 내용
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
            df_result["기사 내용"] = text_result
            page += 1
            df_news = pd.concat([df_news, df_result])
    df_news.to_csv('news_crawling.csv', mode='w', encoding='utf-8-sig')
    return df_news

    # 종목 리스트 파일 열기


def get_price(company_code, date):
    # day_count = "50"
    url = "https://fchart.stock.naver.com/sise.nhn?symbol={}&timeframe=day&count=300&requestType=0".format(
        company_code)
    get_result = requests.get(url)
    bs_obj = BeautifulSoup(get_result.content, "html.parser")

    # information
    inf = bs_obj.select('item')
    columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
    df_inf = pd.DataFrame([], columns=columns, index=range(len(inf)))

    for i in range(len(inf)):
        df_inf.iloc[i] = str(inf[i]['data']).split('|')

    # df_inf['Date'] = datetime.strptime(df_inf['Date'],'%Y%m%d').strftime('%Y.%m.%d')
    df_inf['Date'] = df_inf.Date.apply(lambda x: datetime.datetime.strptime(x, '%Y%m%d').strftime('%Y.%m.%d'))
    code_index = [company_code] * len(df_inf.index)
    df_inf["코드"] = code_index
    df_inf = df_inf.set_index("코드")

    date_datetime = datetime.datetime.strptime(date, '%Y.%m.%d')
    date_before = date_datetime + datetime.timedelta(days=-1)
    date_before = date_before.strftime("%Y.%m.%d")

    df_inf = df_inf.loc[(df_inf["Date"] == date) ^ (df_inf["Date"] == date_before)]

    df_inf.to_csv('company_price.csv', mode='w', encoding='utf-8-sig')
    print(df_inf)


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


    elif bool(str(company) in values) == True:  # Input에 종목코드로 넣었을 때
        company_code = str(company)
        crawler(company_code, maxpage)

    else:
        print("no name or code")


def main():
    # info_main = input("=" * 50 + "\n" + "실시간 뉴스기사 다운받기." + "\n" + " 시작하시려면 Enter를 눌러주세요." + "\n" + "=" * 50)

    company = "삼성전자"
    max_date = "2020.06.15"

    convert_to_code(company, max_date)
def main(company = "005930", crawling_date = "2020.06.18"):
    """
    company에 company_list에 있는 회사 중 하나를 적고,
    crawling_date 날짜를 적으면,
    그 날짜의 그 회사의 뉴스를 네이버 금융에서 크롤링하여 news_crawling.csv에 저장한다.
    :return:
    """
    #company  종목 코드, 005930은 삼성전자
    #crawling_date  뉴스를 가져올 날짜, 연도.월.일 로 적으면 된다.

    if crawling_date == datetime.datetime.today().strftime("%Y.%m.%d"):
        print("It's today, can't get result")
    else:
        convert_to_code(company, crawling_date)


main()
