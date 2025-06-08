import os,json,csv
import pandas as pd

labels={'04379243':["table",0],"04225987":["skateboard",1],"04099429":["rocket,projectile",2],"03948459":["pistol",3],"03797390":["mug",4],"03790512":["motorcycle,bike",5],"03642806":["computer",6],"03636649":["lamp",7],"03624134":["knife",8],"03467517":["guitar",9],"03261776":["earphone,earpiece,headphone,phone",10],"03001627":["chair",11],"02958343":["car,auto,automobile,machine,motorcar",12],"02954340":["cap",13],"02773838":["bag",14],"02691156":["airplane,aeroplane,plane",15]}

with open('python/DensePoint/train.json','r') as f:
    data=json.load(f)

sum=0
print("当前工作目录是：", os.getcwd())
out=[['file','label']]
for u in data:
    cnt=0
    for v in data[u]:
        cnt+=1
        s=u+'\\ply_file\\'+v+'.ply'
        out.append([s,labels[u][1]])
        if(cnt>1):
            print(cnt)
            break

with open('python/DensePoint/train.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerows(out) 

out=pd.read_csv('python/DensePoint/train.csv')
print(out.iloc[0])