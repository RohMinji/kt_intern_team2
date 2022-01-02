import csv
import pandas as pd
from matplotlib import pyplot as plt
from PIL import Image, ImageDraw, ImageFont
import textwrap

#하품 0 자리비움 1
earDF = pd.read_csv("static/data/imgDF.csv")
outDF= pd.read_csv("static/data/txtDF.csv")

earDF.columns=['earRatio','time']
outDF.columns=['label','time']

earDF=earDF.drop_duplicates(['time'])
outDF=outDF.drop_duplicates(['time'])

plt.figure(figsize=(8,8))
plt.title("Sleep Status", fontsize=15)
plt.plot( earDF["time"],earDF["earRatio"])
plt.grid()
plt.xlabel('time')
plt.ylabel('sleepiness')
plt.legend(fontsize=13)
plt.xticks(rotation=90)
plt.savefig('static/img/sleepGraph.png')

fnt = ImageFont.truetype("static/font/malgun.ttf",18)
img = Image.new('RGB', (500, 500),color="white")
pallete = ImageDraw.Draw(img)

lines = textwrap
line_size=0

for idx, row in outDF.iterrows():
    time=row['time']
    if row['label']==0:
        pallete.text((40, 40+line_size),str(time)+ ' 에 하품을 하였습니다.',font=fnt,fill="black")
    else:
        pallete.text((40, 40+line_size),str(time)+ ' 에 자리를 이탈하였습니다.',font=fnt,fill="black")
    line_size+=25

img.save('static/img/text.png')
#img.show()
