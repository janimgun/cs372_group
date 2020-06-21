""" 주식종목 뉴스(네이버 파이넌스) Crawling 하기 """

from bs4 import BeautifulSoup
import requests
import re
import pandas as pd
import os
from datetime import datetime
import datetime


#df_news = pd.DataFrame()

def adv_crawler_part(args):
    def _sub():
        company_code, date,page = args
        url = 'https://finance.naver.com/item/news_news.nhn?code=' + str(company_code) + '&page=' + str(page)
        #print(url)
        try : source_code = requests.get(url, timeout = 10).text
        except: print("timeout", args); return None
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
        # print(date_result)
        date_day = []
        date_time = []
        for i in date_result:
            temp = i.lstrip().split(' ')
            date_day.append(temp[0])
            date_time.append(int(temp[1].split(':')[0]))
    
        # 뉴스 매체
        sources = html.select('.info')
        source_result = [source.get_text() for source in sources]
        company_result = [str(company_code)] * len(date_result)
    
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
        
        pre_R = [I for I in zip(company_result, date_day, date_time, source_result, title_result, link_result, text_result) if I[1] >= date]
        if len(pre_R) == 0: return None 
        '''
        print(page)
        try:print(page, pre_R[-1][1:2])
        except: print(page, pre_R)
        '''
        for i,I in enumerate([company_result, date_day, date_time, source_result, title_result, link_result, text_result]):
            I = [J[i] for J in pre_R]
        result = {"코드": company_result, "날짜": date_day, "시간": date_time, "언론사": source_result, "기사제목": title_result,
                  "링크": link_result, "기사내용": text_result}
        df_result = pd.DataFrame(result)
        return df_result
    try: return _sub()
    except: return None
def crawler(company_code, date):
    last_result = None
    result = []
    out_loop  =25
    in_loop = 20
    import multiprocessing
    if __name__ == 'multiprocessing_news_crawling':
        with multiprocessing.Pool(5) as pool:
            for i in range(out_loop):
                #res = [func(I) for I in  [i*20 + j for j in range(20)]]
                res = [I for I in pool.map(adv_crawler_part, [(company_code, date, i*in_loop + j) for j in range(in_loop)]) if I is not None]
                print("res size = ", len(res))
                if not res or (last_result and str(last_result) == str(res)) : break
                last_result = res
                result.extend(res)
                if len(res) < in_loop / 2: break
    return pd.concat(result)
def get_price(company_code):
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
    '''
    date_datetime = datetime.datetime.strptime(date, '%Y.%m.%d')
    date_before = date_datetime + datetime.timedelta(days=-1)
    date_before = date_before.strftime("%Y.%m.%d")

    df_inf = df_inf.loc[(df_inf["Date"] == date) ^ (df_inf["Date"] == date_before)]
    '''
    return df_inf
    df_inf.to_csv('company_price.csv', mode='w', encoding='utf-8-sig')
    print(df_inf)

