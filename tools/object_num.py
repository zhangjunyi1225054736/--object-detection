#统计训练集中每一类的个数
import json
import csv
with open('/mnt/md126/zhangjunyi/365-object-detection/365_json/objects365_Tiny_train.json', 'r') as f:
    data = json.load(f)

annotations = data['annotations']
category = data['categories']
result = {}
for i in range(len(annotations)):
	id = annotations[i]['category_id']
	if id not in result.keys():
	    result[id] = 1
	else :
		result[id] += 1


print(category)

out = open("object_num.txt","w", newline='')
csv_write = csv.writer(out)
for key in result.keys():
	for i in range(len(category)):
		if category[i]['id'] == key:
			csv_write.writerow([key,category[i]['name'],result[key]])
			break
	
