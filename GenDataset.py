import os 
import json
import numpy as np
PATH=os.path.join(".","Annotations")
files=os.listdir(PATH)
arr=[]
for file in files:
    filename=os.path.join(PATH,file)
    folder=file.replace('annotation','').replace('.json','')
    
    datas=json.load(open(filename))
    
    
    for j in range(len(datas)):
            to_dump={"folder":folder,"an_index":j}
            arr.append(to_dump)
print(len(arr))
with open('./JSON.json','w') as f:
     print("dumping")
     json.dump(arr,f)
            
            
            

