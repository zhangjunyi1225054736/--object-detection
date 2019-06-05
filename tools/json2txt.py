import json

with open('/mnt/md126/zhangjunyi/365-object-detection/365_json/objects365_Tiny_val.json', 'r') as f:
    data = json.load(f)

image = data['images']
annotations = data['annotations']
for i in range(len(image)):
 	name = image[i]['file_name']
 	print("name:", name)
 	id = image[i]['id']
 	count = 1
 	f = open('/mnt/md126/zhangjunyi/365-object-detection/365_txt/'+name[0:23]+'.txt',"a+", newline='')
 	for j in range(len(annotations)):
 		if(annotations[j]['image_id']==id):
 			new_row = annotations[j]['bbox'] 
			
 			new_row.append(1)
 			new_row.append(annotations[j]['category_id'])
 			new_row.append(1)
 			new_row.append(0)
 			
 			new_row = str(new_row).strip('[').strip(']').replace('\'','')+'\n'
 			f.write(new_row)
 			print("count:",count)
 			count = count + 1
