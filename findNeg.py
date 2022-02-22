import json
import os
data=json.load(open("./dataset.json"))
ann=json.load(open("JSON.json"))
neg=[]
for idx,d in enumerate(data):
    track=d["track"]
    folder=ann[idx]["folder"]
    an_index=ann[idx]["an_index"]
    for tr in track:
        if tr[0]<0 or tr[1]<0 or tr[2]<0 or tr[3]<0:
            neg.append((idx,folder,an_index))
            break
print(neg)
