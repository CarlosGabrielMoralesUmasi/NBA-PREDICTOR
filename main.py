import pandas as pd

# Lista de nombres de archivos
file_names = [
    'common_player_info', 'draft_combine_stats', 'draft_history', 
    'game', 'game_info', 'game_summary', 'inactive_players', 
    'line_score', 'officials', 'other_stats', 'play_by_play', 
    'player', 'team', 'team_details', 'team_history', 'team_info_common'
]

# Diccionario para almacenar los DataFrames
data_frames = {}

# Cargar cada archivo y almacenar en el diccionario
for file in file_names:
    data_frames[file] = pd.read_csv(f'C:\\Users\\carlo\\Documents\\Proyectos\\NBA_predict\\csv\\{file}.csv')
    print(f"Resumen de {file}:")
    print(data_frames[file].info())
    print(data_frames[file].head())
    print("\n")

# Ejemplo para acceder a un DataFrame específico
# print(data_frames['common_player_info'].head())
