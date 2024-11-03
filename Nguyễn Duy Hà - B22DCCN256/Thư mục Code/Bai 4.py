#256
from bs4 import BeautifulSoup
import requests
import sys
import pandas as pd
import time
import csv

    #Set mã hóa UTF-8 cho đầu ra
sys.stdout.reconfigure(encoding='utf-8')

if __name__ == "__main__":
    url = 'https://www.footballtransfers.com/us/leagues-cups/national/uk/premier-league/2023-2024'

    #Crawl từ url
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    table = soup.find('table', {
        'class': 'table table-striped table-hover leaguetable mvp-table ranking-table mb-0'
    })

    teams_data = [] 

    if table:
        tbody = table.find('tbody')
        if tbody:
            teams = tbody.find_all('a', href=True)

            for team in teams:
                teams_data.append([team.text.strip(), team['href']])

            print ("Hoàn thành lấy dữ liệu các team")
        else:
            print('Không tìm thấy <tbody>')
    else:
        print('Không tìm thấy bảng dữ liệu')

    
    players_data = []

    for team in teams_data:
        team_name = team[0]
        team_url = team[1]
        print(team_name, team_url)

        r_tmp = requests.get(team_url)
        soup_tmp = BeautifulSoup(r_tmp.text, 'html.parser')
        
        table_tmp = soup_tmp.find('table', {
            'class': 'table table-striped-rowspan ft-table mb-0'
        })

        if  table_tmp:
            tbody_tmp = table_tmp.find('tbody')
            if tbody_tmp:
                players = tbody_tmp.find_all('tr')

                for player in players:
                    if "odd" in player['class'] or "even" in player['class']:
                        player_name = player.find('th').find('span').get_text(strip = True)
                        player_cost = player.find_all('td')[-1].get_text(strip = True)
                        players_data.append([player_name, team_name,  player_cost])
                
                print ("------------Hoàn thành lấy giá các cầu thủ của team: ", team_name, )
            else:
                print('Không tìm thấy <tbody> cầu thủ')
        else:
            print('Không tìm thấy bảng dữ liệu cầu thủ')
        
        time.sleep(4)

    df = pd.DataFrame(players_data, columns=['Player', 'Team', 'Cost'])
    df.to_csv("results4.csv", index=False, encoding='utf-8-sig')

    print("--- Đã lưu thông tin  giá các cầu thủ vào file results4.csv ---")