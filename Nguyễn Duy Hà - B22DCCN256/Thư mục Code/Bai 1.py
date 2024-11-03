#256
from bs4 import BeautifulSoup
import requests
import sys
import pandas as pd
import time
import csv
    #Set mã hóa UTF-8 cho đầu ra
sys.stdout.reconfigure(encoding='utf-8')

    #Hàm xử lý dữ liệu cầu thủ
def Data_Processing_of_Footballer(tmp_tr, team_name, player_data_tmp, mp):
    #information player
    player_name = tmp_tr.find('th', {'data-stat': 'player'}).get_text(strip=True)
    player_national = tmp_tr.find('td', {'data-stat': 'nationality'}).find('a')['href'].split('/')[-1].replace('-Football', ' ') if tmp_tr.find('td', {'data-stat': 'nationality'}).find('a') else "N/a"
    player_position = tmp_tr.find('td', {'data-stat': 'position'}).get_text(strip=True)
    player_age = tmp_tr.find('td', {'data-stat': 'age'}).get_text(strip=True)
    #Playing time
    player_games = tmp_tr.find('td', {'data-stat': 'games'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'games'}).get_text(strip=True) else "N/a"
    player_games_starts = tmp_tr.find('td', {'data-stat': 'games_starts'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'games_starts'}).get_text(strip=True) else "N/a"
    player_minutes = tmp_tr.find('td', {'data-stat': 'minutes'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'minutes'}).get_text(strip=True) else "N/a"

    #Performance
    player_goals_pens = tmp_tr.find('td', {'data-stat': 'goals_pens'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'goals_pens'}).get_text(strip=True) else "N/a"
    player_pens_made = tmp_tr.find('td', {'data-stat': 'pens_made'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'pens_made'}).get_text(strip=True) else "N/a"
    player_assists = tmp_tr.find('td', {'data-stat': 'assists'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'assists'}).get_text(strip=True) else "N/a"
    player_cards_yellow = tmp_tr.find('td', {'data-stat': 'cards_yellow'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'cards_yellow'}).get_text(strip=True) else "N/a"
    player_cards_red = tmp_tr.find('td', {'data-stat': 'cards_red'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'cards_red'}).get_text(strip=True) else "N/a"

    #Expected 
    player_xg = tmp_tr.find('td', {'data-stat': 'xg'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'xg'}).get_text(strip=True) else "N/a"
    player_npxg = tmp_tr.find('td', {'data-stat': 'npxg'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'npxg'}).get_text(strip=True) else "N/a"
    player_xg_assist = tmp_tr.find('td', {'data-stat': 'xg_assist'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'xg_assist'}).get_text(strip=True) else "N/a"

    #Progression
    player_progressive_carries = tmp_tr.find('td', {'data-stat': 'progressive_carries'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'progressive_carries'}).get_text(strip=True) else "N/a"
    player_progressive_passes = tmp_tr.find('td', {'data-stat': 'progressive_passes'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'progressive_passes'}).get_text(strip=True) else "N/a"
    player_progressive_passes_received = tmp_tr.find('td', {'data-stat': 'progressive_passes_received'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'progressive_passes_received'}).get_text(strip=True) else "N/a"

    #Per 90 minutes
    player_goals_per90 = tmp_tr.find('td', {'data-stat': 'goals_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'goals_per90'}).get_text(strip=True) else "N/a"
    player_assists_per90 = tmp_tr.find('td', {'data-stat': 'assists_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'assists_per90'}).get_text(strip=True) else "N/a"
    player_goals_assists_per90 = tmp_tr.find('td', {'data-stat': 'goals_assists_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'goals_assists_per90'}).get_text(strip=True) else "N/a"
    player_goals_pens_per90 = tmp_tr.find('td', {'data-stat': 'goals_pens_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'goals_pens_per90'}).get_text(strip=True) else "N/a"
    player_goals_assists_pens_per90 = tmp_tr.find('td', {'data-stat': 'goals_assists_pens_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'goals_assists_pens_per90'}).get_text(strip=True) else "N/a"
    player_xg_per90 = tmp_tr.find('td', {'data-stat': 'xg_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'xg_per90'}).get_text(strip=True) else "N/a"
    player_xg_assist_per90 = tmp_tr.find('td', {'data-stat': 'xg_assist_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'xg_assist_per90'}).get_text(strip=True) else "N/a"
    player_xg_xg_assist_per90 = tmp_tr.find('td', {'data-stat': 'xg_xg_assist_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'xg_xg_assist_per90'}).get_text(strip=True) else "N/a"
    player_npxg_per90 = tmp_tr.find('td', {'data-stat': 'npxg_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'npxg_per90'}).get_text(strip=True) else "N/a"
    player_npxg_xg_assist_per90 = tmp_tr.find('td', {'data-stat': 'npxg_xg_assist_per90'}).get_text(strip=True) if tmp_tr.find('td', {'data-stat': 'npxg_xg_assist_per90'}).get_text(strip=True) else "N/a"

    
    tmp = [
        player_name, player_national, team_name, player_position, player_age, player_games, player_games_starts, player_minutes, 
        player_goals_pens, player_pens_made, player_assists, player_cards_yellow, player_cards_red, player_xg, 
        player_npxg, player_xg_assist, player_progressive_carries, player_progressive_passes, 
        player_progressive_passes_received, player_goals_per90, player_assists_per90, player_goals_assists_per90, 
        player_goals_pens_per90, player_goals_assists_pens_per90, player_xg_per90, player_xg_assist_per90, player_xg_xg_assist_per90,
        player_npxg_per90, player_npxg_xg_assist_per90
    ]
    player_data_tmp.append(tmp)
    mp[player_name] = player_data_tmp[-1]

    return player_data_tmp

    #Hàm xử lý dữ liệu thủ môn
def Data_Processing_of_Goalkeeper(player):
    player_GA = player.find('td', {'data-stat': 'gk_goals_against'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_goals_against'}).get_text(strip=True) else "N/a"
    player_GA90 = player.find('td', {'data-stat': 'gk_goals_against_per90'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_goals_against_per90'}).get_text(strip=True) else "N/a"
    player_SoTA = player.find('td', {'data-stat': 'gk_shots_on_target_against'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_shots_on_target_against'}).get_text(strip=True) else "N/a"
    player_Saves = player.find('td', {'data-stat': 'gk_saves'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_saves'}).get_text(strip=True) else "N/a"
    player_SaveP = player.find('td', {'data-stat': 'gk_save_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_save_pct'}).get_text(strip=True) else "N/a"
    player_W = player.find('td', {'data-stat': 'gk_wins'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_wins'}).get_text(strip=True) else "N/a"
    player_D = player.find('td', {'data-stat': 'gk_ties'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_ties'}).get_text(strip=True) else "N/a"
    player_L = player.find('td', {'data-stat': 'gk_losses'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_losses'}).get_text(strip=True) else "N/a"
    player_CS = player.find('td', {'data-stat': 'gk_clean_sheets'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_clean_sheets'}).get_text(strip=True) else "N/a"
    player_CSP = player.find('td', {'data-stat': 'gk_clean_sheets_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_clean_sheets_pct'}).get_text(strip=True) else "N/a"
    player_PKatt = player.find('td', {'data-stat': 'gk_pens_att'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_pens_att'}).get_text(strip=True) else "N/a"
    player_PKA = player.find('td', {'data-stat': 'gk_pens_allowed'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_pens_allowed'}).get_text(strip=True) else "N/a"
    player_PKsv = player.find('td', {'data-stat': 'gk_pens_saved'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_pens_saved'}).get_text(strip=True) else "N/a"
    player_PKm = player.find('td', {'data-stat': 'gk_pens_missed'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_pens_missed'}).get_text(strip=True) else "N/a"
    player_SaveP = player.find('td', {'data-stat': 'gk_pens_save_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'gk_pens_save_pct'}).get_text(strip=True) else "N/a"
    
    #Trả về list chỉ số thủ môn
    return [player_GA, player_GA90, player_SoTA, player_Saves, player_SaveP, player_W, player_D, player_L, player_CS, player_CSP, player_PKatt, player_PKA, player_PKsv, player_PKm, player_SaveP]

#Hàm xử lý dữ liệu Shooting
def Data_Processing_of_Shooting(player):
    player_Gls = player.find('td', {'data-stat': 'goals'}).get_text(strip=True) if player.find('td', {'data-stat': 'goals'}).get_text(strip=True) else "N/a"
    player_Sh = player.find('td', {'data-stat': 'shots'}).get_text(strip=True) if player.find('td', {'data-stat': 'shots'}).get_text(strip=True) else "N/a"
    player_SoT = player.find('td', {'data-stat': 'shots_on_target'}).get_text(strip=True) if player.find('td', {'data-stat': 'shots_on_target'}).get_text(strip=True) else "N/a"
    player_SoTP = player.find('td', {'data-stat': 'shots_on_target_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'shots_on_target_pct'}).get_text(strip=True) else "N/a"
    player_Sh90 = player.find('td', {'data-stat': 'shots_per90'}).get_text(strip=True) if player.find('td', {'data-stat': 'shots_per90'}).get_text(strip=True) else "N/a"
    player_Sot90 = player.find('td', {'data-stat': 'shots_on_target_per90'}).get_text(strip=True) if player.find('td', {'data-stat': 'shots_on_target_per90'}).get_text(strip=True) else "N/a"
    player_GSh = player.find('td', {'data-stat': 'goals_per_shot'}).get_text(strip=True) if player.find('td', {'data-stat': 'goals_per_shot'}).get_text(strip=True) else "N/a"
    player_GSoT = player.find('td', {'data-stat': 'goals_per_shot_on_target'}).get_text(strip=True) if player.find('td', {'data-stat': 'goals_per_shot_on_target'}).get_text(strip=True) else "N/a"
    player_Dist = player.find('td', {'data-stat': 'average_shot_distance'}).get_text(strip=True) if player.find('td', {'data-stat': 'average_shot_distance'}).get_text(strip=True) else "N/a"
    player_FK = player.find('td', {'data-stat': 'shots_free_kicks'}).get_text(strip=True) if player.find('td', {'data-stat': 'shots_free_kicks'}).get_text(strip=True) else "N/a"
    player_PK = player.find('td', {'data-stat': 'pens_made'}).get_text(strip=True) if player.find('td', {'data-stat': 'pens_made'}).get_text(strip=True) else "N/a"
    player_PKatt = player.find('td', {'data-stat': 'pens_att'}).get_text(strip=True) if player.find('td', {'data-stat': 'pens_att'}).get_text(strip=True) else "N/a"
    player_xG = player.find('td', {'data-stat': 'xg'}).get_text(strip=True) if player.find('td', {'data-stat': 'xg'}).get_text(strip=True) else "N/a"
    player_npxG = player.find('td', {'data-stat': 'npxg'}).get_text(strip=True) if player.find('td', {'data-stat': 'npxg'}).get_text(strip=True) else "N/a"
    player_npxGSh = player.find('td', {'data-stat': 'npxg_per_shot'}).get_text(strip=True) if player.find('td', {'data-stat': 'npxg_per_shot'}).get_text(strip=True) else "N/a"
    player_GxG = player.find('td', {'data-stat': 'xg_net'}).get_text(strip=True) if player.find('td', {'data-stat': 'xg_net'}).get_text(strip=True) else "N/a"
    player_npGxG = player.find('td', {'data-stat': 'npxg_net'}).get_text(strip=True) if player.find('td', {'data-stat': 'npxg_net'}).get_text(strip=True) else "N/a"

    # Trả về list chỉ số shooting
    return [player_Gls, player_Sh, player_SoT, player_SoTP, player_Sh90, player_Sot90, player_GSh, player_GSoT, player_Dist, player_FK, player_PK, player_PKatt, player_xG, player_npxG, player_npxGSh, player_GxG, player_npGxG]

#Hàm xử lý dữ liệu Passing
def Data_Processing_of_Passing(player):
    player_Cmp = player.find('td', {'data-stat': 'passes_completed'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_completed'}).get_text(strip=True) else "N/a"
    player_Att = player.find('td', {'data-stat': 'passes'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes'}).get_text(strip=True) else "N/a"
    player_CmpP = player.find('td', {'data-stat': 'passes_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_pct'}).get_text(strip=True) else "N/a"
    player_TotDist = player.find('td', {'data-stat': 'passes_total_distance'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_total_distance'}).get_text(strip=True) else "N/a"
    player_PrgDist = player.find('td', {'data-stat': 'passes_progressive_distance'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_progressive_distance'}).get_text(strip=True) else "N/a"
    player_Short_Cmp = player.find('td', {'data-stat': 'passes_completed_short'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_completed_short'}).get_text(strip=True) else "N/a"
    player_Short_Att = player.find('td', {'data-stat': 'passes_short'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_short'}).get_text(strip=True) else "N/a"
    plaer_Short_CmpP = player.find('td', {'data-stat': 'passes_pct_short'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_pct_short'}).get_text(strip=True) else "N/a"
    player_Medium_Cmp = player.find('td', {'data-stat': 'passes_completed_medium'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_completed_medium'}).get_text(strip=True) else "N/a"
    player_Medium_Att = player.find('td', {'data-stat': 'passes_medium'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_medium'}).get_text(strip=True) else "N/a"
    player_Medium_CmpP = player.find('td', {'data-stat': 'passes_pct_medium'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_pct_medium'}).get_text(strip=True) else "N/a"
    player_Long_Cmp = player.find('td', {'data-stat': 'passes_completed_long'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_completed_long'}).get_text(strip=True) else "N/a"
    player_Long_Att = player.find('td', {'data-stat': 'passes_long'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_long'}).get_text(strip=True) else "N/a"
    player_Long_CmpP = player.find('td', {'data-stat': 'passes_pct_long'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_pct_long'}).get_text(strip=True) else "N/a"
    player_Ast = player.find('td', {'data-stat': 'assists'}).get_text(strip=True) if player.find('td', {'data-stat': 'assists'}).get_text(strip=True) else "N/a"
    player_xAG = player.find('td', {'data-stat': 'xg_assist'}).get_text(strip=True) if player.find('td', {'data-stat': 'xg_assist'}).get_text(strip=True) else "N/a"
    player_xA = player.find('td', {'data-stat': 'pass_xa'}).get_text(strip=True) if player.find('td', {'data-stat': 'pass_xa'}).get_text(strip=True) else "N/a"
    palyer_AxAG = player.find('td', {'data-stat': 'xg_assist_net'}).get_text(strip=True) if player.find('td', {'data-stat': 'xg_assist_net'}).get_text(strip=True) else "N/a"
    player_KP = player.find('td', {'data-stat': 'assisted_shots'}).get_text(strip=True) if player.find('td', {'data-stat': 'assisted_shots'}).get_text(strip=True) else "N/a"
    player_1div3 = player.find('td', {'data-stat': 'passes_into_final_third'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_into_final_third'}).get_text(strip=True) else "N/a"
    player_PPA = player.find('td', {'data-stat': 'passes_into_penalty_area'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_into_penalty_area'}).get_text(strip=True) else "N/a"
    player_CrsPA = player.find('td', {'data-stat': 'crosses_into_penalty_area'}).get_text(strip=True) if player.find('td', {'data-stat': 'crosses_into_penalty_area'}).get_text(strip=True) else "N/a"
    player_PrgP = player.find('td', {'data-stat': 'progressive_passes'}).get_text(strip=True) if player.find('td', {'data-stat': 'progressive_passes'}).get_text(strip=True) else "N/a"

    #Trả về list chỉ số passing
    return [player_Cmp, player_Att, player_CmpP, player_TotDist, player_PrgDist, player_Short_Cmp, player_Short_Att, plaer_Short_CmpP, player_Medium_Cmp, player_Medium_Att, player_Medium_CmpP, player_Long_Cmp, player_Long_Att, player_Long_CmpP, player_Ast, player_xAG, player_xA, palyer_AxAG, player_KP, player_1div3, player_PPA, player_CrsPA, player_PrgP]

#Hàm xử lý dữ liệu Pass Types
def Data_Processing_of_Pass_Types(player):
    player_Live = player.find('td', {'data-stat': 'passes_live'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_live'}).get_text(strip=True) else "N/a"
    player_Dead = player.find('td', {'data-stat': 'passes_dead'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_dead'}).get_text(strip=True) else "N/a"
    player_FK = player.find('td', {'data-stat': 'passes_free_kicks'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_free_kicks'}).get_text(strip=True) else "N/a"
    player_TB = player.find('td', {'data-stat': 'through_balls'}).get_text(strip=True) if player.find('td', {'data-stat': 'through_balls'}).get_text(strip=True) else "N/a"
    player_Sw = player.find('td', {'data-stat': 'passes_switches'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_switches'}).get_text(strip=True) else "N/a"
    player_Crs = player.find('td', {'data-stat': 'crosses'}).get_text(strip=True) if player.find('td', {'data-stat': 'crosses'}).get_text(strip=True) else "N/a"
    player_TI = player.find('td', {'data-stat': 'throw_ins'}).get_text(strip=True) if player.find('td', {'data-stat': 'throw_ins'}).get_text(strip=True) else "N/a"
    player_CK = player.find('td', {'data-stat': 'corner_kicks'}).get_text(strip=True) if player.find('td', {'data-stat': 'corner_kicks'}).get_text(strip=True) else "N/a"
    player_In = player.find('td', {'data-stat': 'corner_kicks_in'}).get_text(strip=True) if player.find('td', {'data-stat': 'corner_kicks_in'}).get_text(strip=True) else "N/a"
    player_Out = player.find('td', {'data-stat': 'corner_kicks_out'}).get_text(strip=True) if player.find('td', {'data-stat': 'corner_kicks_out'}).get_text(strip=True) else "N/a"
    player_Str = player.find('td', {'data-stat': 'corner_kicks_straight'}).get_text(strip=True) if player.find('td', {'data-stat': 'corner_kicks_straight'}).get_text(strip=True) else "N/a"
    player_Cmp = player.find('td', {'data-stat': 'passes_completed'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_completed'}).get_text(strip=True) else "N/a"
    player_Off = player.find('td', {'data-stat': 'passes_offsides'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_offsides'}).get_text(strip=True) else "N/a"
    player_Blocks = player.find('td', {'data-stat': 'passes_blocked'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_blocked'}).get_text(strip=True) else "N/a"

    #Trả về list chỉ số pass types
    return [player_Live, player_Dead, player_FK, player_TB, player_Sw, player_Crs, player_TI, player_CK, player_In, player_Out, player_Str, player_Cmp, player_Off, player_Blocks]

#Hàm xử lý dữ liệu Goal and Shot Creation
def Data_Processing_of_Goal_and_Shot_Creation(player):
    player_SCA = player.find('td', {'data-stat': 'sca'}).get_text(strip=True) if player.find('td', {'data-stat': 'sca'}).get_text(strip=True) else "N/a"
    player_SCA90 = player.find('td', {'data-stat': 'sca_per90'}).get_text(strip=True) if player.find('td', {'data-stat': 'sca_per90'}).get_text(strip=True) else "N/a"
    player_SCA_PassLive = player.find('td', {'data-stat': 'sca_passes_live'}).get_text(strip=True) if player.find('td', {'data-stat': 'sca_passes_live'}).get_text(strip=True) else "N/a"
    player_SCA_PassDead = player.find('td', {'data-stat': 'sca_passes_dead'}).get_text(strip=True) if player.find('td', {'data-stat': 'sca_passes_dead'}).get_text(strip=True) else "N/a"
    player_SCA_TO = player.find('td', {'data-stat': 'sca_take_ons'}).get_text(strip=True) if player.find('td', {'data-stat': 'sca_take_ons'}).get_text(strip=True) else "N/a"
    player_SCA_Sh = player.find('td', {'data-stat': 'sca_shots'}).get_text(strip=True) if player.find('td', {'data-stat': 'sca_shots'}).get_text(strip=True) else "N/a"
    player_SCA_Fld = player.find('td', {'data-stat': 'sca_fouled'}).get_text(strip=True) if player.find('td', {'data-stat': 'sca_fouled'}).get_text(strip=True) else "N/a"
    player_SCA_Def = player.find('td', {'data-stat': 'sca_defense'}).get_text(strip=True) if player.find('td', {'data-stat': 'sca_defense'}).get_text(strip=True) else "N/a"
    player_GCA = player.find('td', {'data-stat': 'gca'}).get_text(strip=True) if player.find('td', {'data-stat': 'gca'}).get_text(strip=True) else "N/a"
    player_GCA90 = player.find('td', {'data-stat': 'gca_per90'}).get_text(strip=True) if player.find('td', {'data-stat': 'gca_per90'}).get_text(strip=True) else "N/a"
    player_GCA_PassLive = player.find('td', {'data-stat': 'gca_passes_live'}).get_text(strip=True) if player.find('td', {'data-stat': 'gca_passes_live'}).get_text(strip=True) else "N/a"
    player_GCA_PassDead = player.find('td', {'data-stat': 'gca_passes_dead'}).get_text(strip=True) if player.find('td', {'data-stat': 'gca_passes_dead'}).get_text(strip=True) else "N/a"
    player_GCA_TO = player.find('td', {'data-stat': 'gca_take_ons'}).get_text(strip=True) if player.find('td', {'data-stat': 'gca_take_ons'}).get_text(strip=True) else "N/a"
    player_GCA_Sh = player.find('td', {'data-stat': 'gca_shots'}).get_text(strip=True) if player.find('td', {'data-stat': 'gca_shots'}).get_text(strip=True) else "N/a"
    player_GCA_Fld = player.find('td', {'data-stat': 'gca_fouled'}).get_text(strip=True) if player.find('td', {'data-stat': 'gca_fouled'}).get_text(strip=True) else "N/a"
    player_GCA_Def = player.find('td', {'data-stat': 'gca_defense'}).get_text(strip=True) if player.find('td', {'data-stat': 'gca_defense'}).get_text(strip=True) else "N/a"

    #Trả về list chỉ số Goal and Shot Creation
    return [player_SCA, player_SCA90, player_SCA_PassLive, player_SCA_PassDead, player_SCA_TO, player_SCA_Sh, player_SCA_Fld, player_SCA_Def, player_GCA, player_GCA90, player_GCA_PassLive, player_GCA_PassDead, player_GCA_TO, player_GCA_Sh, player_GCA_Fld, player_GCA_Def]

#Hàm xử lý dữ liệu Defensive Actions
def Data_Processing_of_Defensive_Actions(player):
    player_Tackes_Tkl = player.find('td', {'data-stat': 'tackles'}).get_text(strip=True) if player.find('td', {'data-stat': 'tackles'}).get_text(strip=True) else "N/a"
    player_Tackes_TklW = player.find('td', {'data-stat': 'tackles_won'}).get_text(strip=True) if player.find('td', {'data-stat': 'tackles_won'}).get_text(strip=True) else "N/a"
    player_Tackes_Def3rd = player.find('td', {'data-stat': 'tackles_def_3rd'}).get_text(strip=True) if player.find('td', {'data-stat': 'tackles_def_3rd'}).get_text(strip=True) else "N/a"
    player_Tackes_Mid3rd = player.find('td', {'data-stat': 'tackles_mid_3rd'}).get_text(strip=True) if player.find('td', {'data-stat': 'tackles_mid_3rd'}).get_text(strip=True) else "N/a"
    player_Tackes_Att3rd = player.find('td', {'data-stat': 'tackles_att_3rd'}).get_text(strip=True) if player.find('td', {'data-stat': 'tackles_att_3rd'}).get_text(strip=True) else "N/a"
    player_Challenges_Tkl = player.find('td', {'data-stat': 'challenge_tackles'}).get_text(strip=True) if player.find('td', {'data-stat': 'challenge_tackles'}).get_text(strip=True) else "N/a"
    player_Challenges_Att = player.find('td', {'data-stat': 'challenges'}).get_text(strip=True) if player.find('td', {'data-stat': 'challenges'}).get_text(strip=True) else "N/a"
    player_Challenges_TklP = player.find('td', {'data-stat': 'challenge_tackles_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'challenge_tackles_pct'}).get_text(strip=True) else "N/a"
    palyer_Challenges_Lost = player.find('td', {'data-stat': 'challenges_lost'}).get_text(strip=True) if player.find('td', {'data-stat': 'challenges_lost'}).get_text(strip=True) else "N/a"
    player_Blocks = player.find('td', {'data-stat': 'blocks'}).get_text(strip=True) if player.find('td', {'data-stat': 'blocks'}).get_text(strip=True) else "N/a"
    player_Blocks_Sh = player.find('td', {'data-stat': 'blocked_shots'}).get_text(strip=True) if player.find('td', {'data-stat': 'blocked_shots'}).get_text(strip=True) else "N/a"
    player_Blocks_Pass = player.find('td', {'data-stat': 'blocked_passes'}).get_text(strip=True) if player.find('td', {'data-stat': 'blocked_passes'}).get_text(strip=True) else "N/a"
    player_Int = player.find('td', {'data-stat': 'interceptions'}).get_text(strip=True) if player.find('td', {'data-stat': 'interceptions'}).get_text(strip=True) else "N/a"
    player_Tkl_Int = player.find('td', {'data-stat': 'tackles_interceptions'}).get_text(strip=True) if player.find('td', {'data-stat': 'tackles_interceptions'}).get_text(strip=True) else "N/a"
    player_Clr = player.find('td', {'data-stat': 'clearances'}).get_text(strip=True) if player.find('td', {'data-stat': 'clearances'}).get_text(strip=True) else "N/a"
    player_Err = player.find('td', {'data-stat': 'errors'}).get_text(strip=True) if player.find('td', {'data-stat': 'errors'}).get_text(strip=True) else "N/a"

    return [player_Tackes_Tkl, player_Tackes_TklW, player_Tackes_Def3rd, player_Tackes_Mid3rd, player_Tackes_Att3rd, player_Challenges_Tkl, player_Challenges_Att, player_Challenges_TklP, palyer_Challenges_Lost, player_Blocks, player_Blocks_Sh, player_Blocks_Pass, player_Int, player_Tkl_Int, player_Clr, player_Err]


def Data_Processing_of_Possession(player):
    player_Touches = player.find('td', {'data-stat': 'touches'}).get_text(strip=True) if player.find('td', {'data-stat': 'touches'}).get_text(strip=True) else "N/a"
    player_DefPen = player.find('td', {'data-stat': 'touches_def_pen_area'}).get_text(strip=True) if player.find('td', {'data-stat': 'touches_def_pen_area'}).get_text(strip=True) else "N/a"
    player_Def3rd = player.find('td', {'data-stat': 'touches_def_3rd'}).get_text(strip=True) if player.find('td', {'data-stat': 'touches_def_3rd'}).get_text(strip=True) else "N/a"
    player_Mid3rd = player.find('td', {'data-stat': 'touches_mid_3rd'}).get_text(strip=True) if player.find('td', {'data-stat': 'touches_mid_3rd'}).get_text(strip=True) else "N/a"
    player_Att3rd = player.find('td', {'data-stat': 'touches_att_3rd'}).get_text(strip=True) if player.find('td', {'data-stat': 'touches_att_3rd'}).get_text(strip=True) else "N/a"
    player_AttPen = player.find('td', {'data-stat': 'touches_att_pen_area'}).get_text(strip=True) if player.find('td', {'data-stat': 'touches_att_pen_area'}).get_text(strip=True) else "N/a"
    player_Touches_Live = player.find('td', {'data-stat': 'touches_live_ball'}).get_text(strip=True) if player.find('td', {'data-stat': 'touches_live_ball'}).get_text(strip=True) else "N/a"
    player_Att = player.find('td', {'data-stat': 'take_ons'}).get_text(strip=True) if player.find('td', {'data-stat': 'take_ons'}).get_text(strip=True) else "N/a"
    player_Succ = player.find('td', {'data-stat': 'take_ons_won'}).get_text(strip=True) if player.find('td', {'data-stat': 'take_ons_won'}).get_text(strip=True) else "N/a"
    player_SuccP = player.find('td', {'data-stat': 'take_ons_won_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'take_ons_won_pct'}).get_text(strip=True) else "N/a"
    player_Tkld = player.find('td', {'data-stat': 'take_ons_tackled'}).get_text(strip=True) if player.find('td', {'data-stat': 'take_ons_tackled'}).get_text(strip=True) else "N/a"
    player_TkldP = player.find('td', {'data-stat': 'take_ons_tackled_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'take_ons_tackled_pct'}).get_text(strip=True) else "N/a"
    player_Carries = player.find('td', {'data-stat': 'carries'}).get_text(strip=True) if player.find('td', {'data-stat': 'carries'}).get_text(strip=True) else "N/a"
    player_TotDist = player.find('td', {'data-stat': 'carries_distance'}).get_text(strip=True) if player.find('td', {'data-stat': 'carries_distance'}).get_text(strip=True) else "N/a"
    player_PrgDist = player.find('td', {'data-stat': 'carries_progressive_distance'}).get_text(strip=True) if player.find('td', {'data-stat': 'carries_progressive_distance'}).get_text(strip=True) else "N/a"
    player_PrgCarries = player.find('td', {'data-stat': 'progressive_carries'}).get_text(strip=True) if player.find('td', {'data-stat': 'progressive_carries'}).get_text(strip=True) else "N/a"
    player_1div3 = player.find('td', {'data-stat': 'carries_into_final_third'}).get_text(strip=True) if player.find('td', {'data-stat': 'carries_into_final_third'}).get_text(strip=True) else "N/a"
    player_CPA = player.find('td', {'data-stat': 'carries_into_penalty_area'}).get_text(strip=True) if player.find('td', {'data-stat': 'carries_into_penalty_area'}).get_text(strip=True) else "N/a"
    player_Mis = player.find('td', {'data-stat': 'miscontrols'}).get_text(strip=True) if player.find('td', {'data-stat': 'miscontrols'}).get_text(strip=True) else "N/a"
    player_Dis = player.find('td', {'data-stat': 'dispossessed'}).get_text(strip=True) if player.find('td', {'data-stat': 'dispossessed'}).get_text(strip=True) else "N/a"
    player_Rec = player.find('td', {'data-stat': 'passes_received'}).get_text(strip=True) if player.find('td', {'data-stat': 'passes_received'}).get_text(strip=True) else "N/a"
    player_PrgR = player.find('td', {'data-stat': 'progressive_passes_received'}).get_text(strip=True) if player.find('td', {'data-stat': 'progressive_passes_received'}).get_text(strip=True) else "N/a"

    #Trả về list chỉ số Possession
    return [player_Touches, player_DefPen, player_Def3rd, player_Mid3rd, player_Att3rd, player_AttPen, player_Touches_Live, player_Att, player_Succ, player_SuccP, player_Tkld, player_TkldP, player_Carries, player_TotDist, player_PrgDist, player_PrgCarries, player_1div3, player_CPA, player_Mis, player_Dis, player_Rec, player_PrgR]

#Hàm xử lý dữ liệu Playing Time
def Data_Processing_of_Playing_Time(player):
    player_Starts = player.find('td', {'data-stat': 'games_starts'}).get_text(strip=True) if player.find('td', {'data-stat': 'games_starts'}).get_text(strip=True) else "N/a"
    player_MinStart = player.find('td', {'data-stat': 'minutes_per_start'}).get_text(strip=True) if player.find('td', {'data-stat': 'minutes_per_start'}).get_text(strip=True) else "N/a"
    player_Compl = player.find('td', {'data-stat': 'games_complete'}).get_text(strip=True) if player.find('td', {'data-stat': 'games_complete'}).get_text(strip=True) else "N/a"
    player_Subs = player.find('td', {'data-stat': 'games_subs'}).get_text(strip=True) if player.find('td', {'data-stat': 'games_subs'}).get_text(strip=True) else "N/a"
    player_MnStart = player.find('td', {'data-stat': 'minutes_per_sub'}).get_text(strip=True) if player.find('td', {'data-stat': 'minutes_per_sub'}).get_text(strip=True) else "N/a"
    player_unSub = player.find('td', {'data-stat': 'unused_subs'}).get_text(strip=True) if player.find('td', {'data-stat': 'unused_subs'}).get_text(strip=True) else "N/a"
    player_PPM = player.find('td', {'data-stat': 'points_per_game'}).get_text(strip=True) if player.find('td', {'data-stat': 'points_per_game'}).get_text(strip=True) else "N/a"
    player_onG = player.find('td', {'data-stat': 'on_goals_for'}).get_text(strip=True) if player.find('td', {'data-stat': 'on_goals_for'}).get_text(strip=True) else "N/a"
    player_onGA = player.find('td', {'data-stat': 'on_goals_against'}).get_text(strip=True) if player.find('td', {'data-stat': 'on_goals_against'}).get_text(strip=True) else "N/a"
    player_onxG = player.find('td', {'data-stat': 'on_xg_for'}).get_text(strip=True) if player.find('td', {'data-stat': 'on_xg_for'}).get_text(strip=True) else "N/a"
    player_onxGA = player.find('td', {'data-stat': 'on_xg_against'}).get_text(strip=True) if player.find('td', {'data-stat': 'on_xg_against'}).get_text(strip=True) else "N/a"

    #Trả về list chỉ số Playing Time
    return [player_Starts, player_MinStart, player_Compl, player_Subs, player_MnStart, player_unSub, player_PPM, player_onG, player_onGA, player_onxG, player_onxGA]

#Hàm xử lý dữ liệu Miscellaneous Stats
def Data_Processing_of_Miscellaneous_Stats(player):
    player_Fls = player.find('td', {'data-stat': 'fouls'}).get_text(strip=True) if player.find('td', {'data-stat': 'fouls'}).get_text(strip=True) else "N/a"
    player_Fld = player.find('td', {'data-stat': 'fouled'}).get_text(strip=True) if player.find('td', {'data-stat': 'fouled'}).get_text(strip=True) else "N/a"
    player_Off = player.find('td', {'data-stat': 'offsides'}).get_text(strip=True) if player.find('td', {'data-stat': 'offsides'}).get_text(strip=True) else "N/a"
    player_Crs = player.find('td', {'data-stat': 'crosses'}).get_text(strip=True) if player.find('td', {'data-stat': 'crosses'}).get_text(strip=True) else "N/a"
    player_OG = player.find('td', {'data-stat': 'own_goals'}).get_text(strip=True) if player.find('td', {'data-stat': 'own_goals'}).get_text(strip=True) else "N/a"
    player_Recov = player.find('td', {'data-stat': 'ball_recoveries'}).get_text(strip=True) if player.find('td', {'data-stat': 'ball_recoveries'}).get_text(strip=True) else "N/a"
    player_Won = player.find('td', {'data-stat': 'aerials_won'}).get_text(strip=True) if player.find('td', {'data-stat': 'aerials_won'}).get_text(strip=True) else "N/a"
    player_Lost = player.find('td', {'data-stat': 'aerials_lost'}).get_text(strip=True) if player.find('td', {'data-stat': 'aerials_lost'}).get_text(strip=True) else "N/a"
    player_WonP = player.find('td', {'data-stat': 'aerials_won_pct'}).get_text(strip=True) if player.find('td', {'data-stat': 'aerials_won_pct'}).get_text(strip=True) else "N/a"

    #Trả về list chỉ số Miscellaneous Stats
    return [player_Fls, player_Fld, player_Off, player_Crs, player_OG, player_Recov, player_Won, player_Lost, player_WonP]


#Hàm cào dữ liệu lấy thông tin cầu thủ của từng đội bóng
def Crawl_Data_For_Each_Team(players_data, team_data):
    #Get set-inf cầu thủ của mỗi đội
    for team in team_data:
        team_name = team[0]
        team_url = team[1]

        print(f"--- Đang cào dữ liệu cầu thủ của đội {team_name} ")
        #Crawl url của từng đội bóng
        r_tmp = requests.get(team_url)
        soup_tmp = BeautifulSoup(r_tmp.content, 'html.parser')

        #Danh sách tạm thời chứa thông tin tất cả cầu thủ của đội bóng hiện tại
        player_data_tmp = []
        mp = {} 

        # Tìm bảng chứa thông tin các cầu thủ
        player_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_standard_9'
        })
        if player_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = player_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                for player in players:
                    player_minutes_matches = player.find('td', {'data-stat': 'minutes'}).get_text(strip=True) if player.find('td', {'data-stat': 'minutes'}).get_text(strip=True) else "N/a"
                    # Lọc ra những cầu thủ đã thi đấu ít nhất 90 phút
                    if player_minutes_matches == "N/a" or int(player_minutes_matches.replace(',','')) < 90: 
                        continue
                    player_data_tmp = Data_Processing_of_Footballer(player, team_name, player_data_tmp, mp) 
            else:
                print(f"- Không tìm thấy thẻ <tbody> trong bảng cầu thủ -")
        else:
            print(f"- Không tìm thấy thẻ <table> chứa cầu thủ trong trang của đội {team_name}. - ")
            
        #Tìm bảng chứa thông tin các thủ môn
        Goalkeeper_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_keeper_9'
        })
        if Goalkeeper_table:
            #Tìm thẻ <tbody> trong <table>
            tbody = Goalkeeper_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                #Danh sách lưu tạm thời tên các thủ môn
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True) 
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Goalkeeper(player)
                        list_tmp.append(player_name)

                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 15
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng thủ môn.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thủ môn trong trang của đội {team_name}.")

        # Tìm bảng chứa thông tin Shooting của các cầu thủ
        Shooting_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_shooting_9'
        })
        if Shooting_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = Shooting_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True)
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Shooting(player)
                        list_tmp.append(player_name)

                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 17
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng Shooting.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thông tin Shooting trong trang của đội {team_name}.")

        # Tìm bảng chứa thông tin Passing của các cầu thủ
        Passing_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_passing_9'
        })
        if Passing_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = Passing_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                # Danh sách lưu tạm thời tên các cầu thủ
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True)
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Passing(player)
                        list_tmp.append(player_name)
                
                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 23
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng Passing.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thông tin Passing trong trang của đội {team_name}.")

        # Tìm bảng chứa thông tin Pass Types của các cầu thủ
        Pass_Types_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_passing_types_9'
        })
        if Pass_Types_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = Pass_Types_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                # Danh sách lưu tạm thời tên các cầu thủ
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True)
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Pass_Types(player)
                        list_tmp.append(player_name)
                
                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 14
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng Pass Types.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thông tin Pass Types trong trang của đội {team_name}.")

        # Tìm bảng chứa thông tin Goal and Shot Creation của các cầu thủ
        Goal_Shot_Creation_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_gca_9'
        })
        if Goal_Shot_Creation_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = Goal_Shot_Creation_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                # Danh sách lưu tạm thời tên các cầu thủ
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True)
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Goal_and_Shot_Creation(player)
                        list_tmp.append(player_name)
                
                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 16
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng Goal and Shot Creation.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thông tin Goal and Shot Creation trong trang của đội {team_name}.")

        # Tìm bảng chứa thông tin Defensive Actions của các cầu thủ
        Defensive_Actions_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_defense_9'
        })
        if Defensive_Actions_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = Defensive_Actions_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                # Danh sách lưu tạm thời tên các cầu thủ
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True)
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Defensive_Actions(player)
                        list_tmp.append(player_name)
                
                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 16
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng Defensive Actions.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thông tin Defensive Actions trong trang của đội {team_name}.")

        # Tìm bảng chứa thông tin Possession của các cầu thủ
        Possession_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_possession_9'
        })
        if Possession_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = Possession_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                # Danh sách lưu tạm thời tên các cầu thủ
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True)
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Possession(player)
                        list_tmp.append(player_name)
                
                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 22
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng Possession.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thông tin Possession trong trang của đội {team_name}.")

        # Tìm bảng chứa thông tin Playing Time của các cầu thủ
        Playing_Time_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_playing_time_9'
        })
        if Playing_Time_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = Playing_Time_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                # Danh sách lưu tạm thời tên các cầu thủ
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True)
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Playing_Time(player)
                        list_tmp.append(player_name)
                
                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 11
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng Playing Time.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thông tin Playing Time trong trang của đội {team_name}.")

        # Tìm bảng chứa thông tin Miscellaneous Stats của các cầu thủ
        Miscellaneous_Stats_table = soup_tmp.find('table', {
            'class': 'stats_table sortable min_width',
            'id': 'stats_misc_9'
        })
        if Miscellaneous_Stats_table:
            # Tìm thẻ <tbody> trong <table>
            tbody = Miscellaneous_Stats_table.find('tbody')
            if tbody:
                players = tbody.find_all('tr')
                # Danh sách lưu tạm thời tên các cầu thủ
                list_tmp = []

                for player in players:
                    player_name = player.find('th', {'data-stat': 'player'}).get_text(strip=True)
                    if player_name in mp:
                        mp[player_name] += Data_Processing_of_Miscellaneous_Stats(player)
                        list_tmp.append(player_name)
                
                for player in player_data_tmp:
                    if player[0] not in list_tmp:
                        player += ["N/a"] * 9
            else:
                print(f"Không tìm thấy thẻ <tbody> bảng Miscellaneous Stats.")
        else:
            print(f"Không tìm thấy thẻ <table> chứa thông tin Miscellaneous Stats trong trang của đội {team_name}.")
        

        # Thêm dữ liệu các cầu thủ của đội bóng vào danh sách chứa dữ liệu của tất cả các cầu thủ
        players_data += player_data_tmp 
        print(f"--- Đã cào xong dữ liệu cầu thủ của đội {team_name}. ")

        # Tạm nghỉ trước khi cào đội tiếp theo
        time.sleep(8)

    return players_data

if __name__ == "__main__":
    # URL to fetch
    url = 'https://fbref.com/en/comps/9/2023-2024/2023-2024-Premier-League-Stats'
    r = requests.get(url)
    soup = BeautifulSoup(r.content, 'html.parser')

    # Tìm bảng chứa thông tin các đội bóng trong mùa giải 2023-2024
    table = soup.find('table', {
        'class': 'stats_table sortable min_width force_mobilize',
        'id': 'results2023-202491_overall'
    })

     # Danh sách chứa dữ liệu đội bóng và URL
    team_data = []

    if table:
        # Tìm thẻ <tbody> trong <table>
        tbody = table.find('tbody')
        if tbody:
            # Lấy tất cả các thẻ <a> có định dạng như yêu cầu trong <tbody>
            teams = tbody.find_all('a', href=True)

            for team in teams:
                if "squads" in team['href']:  # Kiểm tra nếu "squads" có trong href
                    team_name = team.get_text(strip=True)
                    team_url = "https://fbref.com" + team['href']
                    team_data.append([team_name, team_url])
                    
            print("--- Danh sách các đội bóng đã được lấy thành công.")
        else:
            print("Không tìm thấy thẻ <tbody>.")
    else:
        print("Không tìm thấy thẻ <table>.")

    #  Danh sach chứa từng cầu thủ của đội bóng
    players_data = []
    players_data = Crawl_Data_For_Each_Team(players_data, team_data)
    
    # Sắp xếp dữ liệu theo first name và tuổi giảm dần
    players_data = sorted(players_data, key=lambda x: (x[0].split()[0], -int(x[4])))


    # Chuyển dữ liệu thành DataFrame và lưu thành file CSV
    df_players = pd.DataFrame(players_data, columns=["Player Name", "Nation", "Team", "Position", "Age", "Matches Played", "Starts", "Minutes", "Non-Penalty Goals", "Penalties Made", "Assists", "Yellow Cards", "Red Cards", "xG", "npxG", "xAG", "PrgC", "PrgP", "PrgR","Gls/90", "Ast/90", "G+A/90", "G-PK/90", "G+A-PK/90", "xG/90", "xAG/90","xG+xAG/90", "npxG/90", "npxG+xAG/90",
                                                     "Goalkeeping_GA", "GGoalkeeping_GA90", "Goalkeeping_SoTA", "Goalkeeping_Saves", "Goalkeeping_Save%", "Goalkeeping_W", "Goalkeeping_D", "Goalkeeping_L", "Goalkeeping_CS", "Goalkeeping_CS%", "Goalkeeping_PKatt", "Goalkeeping_PKA", "Goalkeeping_Pksv", "Goalkeeping_PKm", "Goalkeeping_Save%",
                                                     "Shooting_Gls", "Shooting_Sh", "Shooting_SoT", "Shooting_SoT%", "Shooting_Sh/90", "Shooting_SoT/90", "Shooting_G/Sh", "Shooting_G/SoT", "Shooting_Dist", "Shooting_FK", "Shooting_PK", "Shooting_PKatt", "Shooting_xG", "Shooting_npxG", "Shooting_npxG/Sh", "Shooting_G-xG", "Shooting_np:G-xG",
                                                     "Passing_Cmp", "Passing_Att", "Passing_Cmp%", "Passing_ToDist", "Passing_PrgDist", "Passing_Short_Cmp", "Passing_Short_Att", "Passing_Short_Cmp%", "Passing_Med_Cmp", "Passing_Med_Att", "Passing_Med_Cmp%", "Passing_Long_Cmp", "Passing_Long_Att", "Passing_Long_Cmp%", "Passing_Ast", "Passing_xAG", "Passing_xA", "Passing_A-xAG", "Passing_KP", "Passing_1/3", "Passing_PPA", "Passing_CrsPA", "Passing_PrgP",
                                                     "Pass_Types_Live", "Pass_Types_Dead", "Pass_Types_FK", "Pass_Types_TB", "Pass_Types_Sw", "Pass_Types_Crs", "Pass_Types_TI", "Pass_Types_CK", "Pass_Types_In", "Pass_Types_Out", "Pass_Types_Str", "Pass_Types_Gmp", "Pass_Types_Off", "Pass_Types_Blocks",
                                                     "GSCreation_SCA", "GSCreation_SCA90", "GGSCreation_SCA_PassLive", "GSCreation_SCA_PassDead", "GSCreation_SCA_TO", "GSCreation_SCA_Sh", "GSCreation_SCA_Fld", "GSCreation_SCA_Def", "GSCreation_GCA", "GSCreation_GCA90", "GSCreation_GCA_PassLive", "GSCreation_GCA_PassDead", "GSCreation_GCA_TO", "GSCreation_GCA_Sh", "GSCreation_GCA_Fld", "GSCreation_GCA_Def",
                                                     "DActions_Tkl", "DActions_TklW", "DActions_Def3rd", "DActions_Mid3rd", "DActions_Att3rd", "DActions_Challenges_Tkl", "DActions_Challenges_Att", "DActions_Challenges_Tkl%", "DActions_Challenges_Lost", "DActions_Blocks", "DActions_Blocks_Sh", "DActions_Blocks_Pass", "DActions_Int", "DActions_Tkl+Int", "DActions_Clr", "DActions_Err",
                                                     "Possession_Touches", "Possession_Def Pen", "Possession_Def 3rd", "Possession_Mid 3rd", "Possession_Att 3rd", "Possession_Att Pen", "Possession_Live", "Possession_Att", "Possession_Succ", "Possession_Succ%", "Possession_Tkld", "Possession_Tkld%", "Possession_Carries", "Possession_TotDist", "Possession_PrgDist", "Possession_PrgC", "Possession_1/3", "Possession_CPA", "Possession_Mis", "Possession_Dis", "Possession_Rec", "Possession_PrgR",
                                                     "PTime_Starts", "PTime_Mn/Start", "PTime_Compl", "PTime_Subs", "PTime_Mn/Sub", "PTime_unSub", "PTime_PPM", "PTime_onG", "PTime_onGA", "PTime_onxG", "PTime_onxGA",
                                                     "MStats_Fls", "MStats_Fld", "MStats_Off", "MStats_Crs", "MStats_OG", "MStats_Recov", "MStats_Won", "MStats_Lost", "MStats_Won%"
                                                     ])
    df_players.to_csv("results.csv", index=False, encoding='utf-8-sig')
    print("----------------- Đã lưu thông tin các cầu thủ vào file results.csv -------------------- ")
