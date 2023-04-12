import csv
import numpy as np 
import pandas as pd 

def read_csv(file_path):
	df=pd.read_csv(file_path)
	data_dict = df['content'].to_dict()
	inv_map = {v: k for k, v in data_dict.items()}
	return inv_map

def read_xls(file_path, sheet_name, write_file='./test.csv', inv_map1=None, inv_map2=None):
	df=pd.read_excel(file_path,sheet_name=sheet_name, header=None)
	with open(write_file, 'a+', encoding='utf-8', newline='') as f:
		for val in df.values:
			if pd.isna(val[1]):
				continue
			if val[0].strip() in inv_map1 and val[1].strip() in inv_map2:
				csv.writer(f, dialect="excel").writerow((inv_map1[val[0].strip()], inv_map2[val[1].strip()]))


if __name__ == "__main__":
	alexa_map = read_csv('alexa_node.csv')
	ifttt_map = read_csv('ifttt_node.csv')
	smartthings_map = read_csv('smartthings_node.csv')
	read_xls("./trigger-action.xls", "ifttt-smartthings", write_file='./ifttt_interact_smartthings.csv', inv_map1=ifttt_map, inv_map2=smartthings_map)
	read_xls("./trigger-action.xls", "alexa-smartthings", write_file='./alexa_interact_smartthings.csv', inv_map1=alexa_map, inv_map2=smartthings_map)
	read_xls("./trigger-action.xls", "alexa-ifttt", write_file='./alexa_interact_ifttt.csv', inv_map1=alexa_map, inv_map2=ifttt_map)

