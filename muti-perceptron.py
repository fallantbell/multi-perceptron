import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np
import math
import random
import matplotlib.animation as animation
from tkinter import filedialog
window=tk.Tk()

def browse_file():
    global filename
    fname = filedialog.askopenfilename(filetypes = (("Template files", "*.txt"), ("All files", "*")))
    filename.set(fname)

listxy=[]
listtmp=[]
listexp=[]
maxaccurate=0 #正確率

def readfile(): #讀檔
    global filename,listxy,listtmp,listexp
    listxy=[]
    listtmp=[]
    listexp=[]
    f=open(filename.get(),"r")
    words=f.read().replace('\r'," ").replace('\n'," ")
    f.close()
    
    words=words.split(" ")
    for i in range(len(words)):
        if i%3==0:
            listtmp.append(float(words[i]))
        elif i%3==1:
            listtmp.append(float(words[i]))
        else:
            listexp.append(int(words[i])%2)
            listxy.append(listtmp)
            listtmp=[]

def activefunction(v):
    return 1/(1+math.exp((-1)*v))

def submit():
    global hiddenlayertext,hiddenpointtext,learingtext,learnnumbertext,listxy,listexp,maxaccurate
    readfile()
    maxaccurate=0

    hl=hiddenlayertext.get()
    hl=int(hl)
    hp=hiddenpointtext.get()
    hp=int(hp)
    rate=learingtext.get() #學習率
    rate=float(rate)
    number=learnnumbertext.get() #學習次數
    number=int(number)

    #======================初始鍵結值==============================

    w=[]
    for i in range(hl):
        w2=[] #每一層神經元
        for j in range(hp):
            w3=[] #每一個神經元
            if i==0: #第一層隱藏層 每一個神經元的鍵結值數量=輸入
                for k in range(3):
                    w3.append(random.uniform(-1,1))
            else: #其他層隱藏層 神經元鑑結值數量=隱藏層每層神經元的數量
                for k in range(hp+1):
                    w3.append(random.uniform(-1,1))
            w2.append(w3)
        w.append(w2)
    w2=[]
    w3=[]
    for k in range(hp+1):
        w3.append(random.uniform(-1,1))
    w2.append(w3)
    w.append(w2)

    #測試

    # w[0][0]=[-1.2,1,1]  
    # w[0][1]=[0.3,1,1]
    # w[1][0]=[0.5,0.4,0.8]

    answ=[]

    #======================訓練========================================

    
    for q in range(number):
        
        note=[]
        for i in range(hl):
            note2=[]
            for j in range(hp):
                note2.append(0)
            note.append(note2)

        #==前饋==
        
        i=q%len(listxy)
        y=[]
        tmp=[-1]
        for j in range(2):
            tmp.append(listxy[i][j])
        y.append(tmp) #y[0] 會和 w[0]那一層的神經元內積

        for j in range(hl): #層
            tmp=[-1]
            for k in range(hp): #神經元
                v=0
                for l in range(len(y[j])): #神經元內的向量
                   v+=w[j][k][l]*y[j][l] #內積出v

                tmp.append(activefunction(v)) #經過活化函數
            y.append(tmp)
        
        #==輸出層==    
        v=0
        for k in range(len(y[hl])):
            v+=w[hl][0][k]*y[hl][k]
        output=activefunction(v)
        
        # if output<0.5 and listexp[i]==0:
        #     continue
        # if output>=0.5 and listexp[i]==1:
        #     continue

        #結果不正確

        note2=[]
        note2.append(output*(1-output)*(listexp[i]-output))
        note.append(note2)

        #==倒傳遞===
        for j in range(hl-1,-1,-1): #層
            for k in range(hp): #神經元
                notetmp=0
                for l in range(len(w[j+1])): #下一層的神經元
                    notetmp+=w[j+1][l][k+1]*note[j+1][l]
                # print(f"j={j} k={k}")
                note[j][k]=notetmp*y[j+1][k+1]*(1-y[j+1][k+1])
        
        # wchallanger=w #wchallanger是新的鍵結值 用來挑戰w的正確率 然後最後要選正確率高的

        # #==更改鍵結值==
        # for j in range(hl):
        #     for k in range(hp):
        #         for l in range(len(y[j])):
        #             wchallanger[j][k][l]=w[j][k][l]+rate*y[j][l]*note[j][k]
        
        # for k in range(len(y[hl])):
        #     wchallanger[hl][0][k]=w[hl][0][k]+rate*y[hl][k]*note[hl][0]
        #wchallanger 的正確率高於原本的w 所以更新w

        for j in range(hl):
            for k in range(hp):
                for l in range(len(y[j])):
                    w[j][k][l]=w[j][k][l]+rate*y[j][l]*note[j][k]
        
        for k in range(len(y[hl])):
            w[hl][0][k]=w[hl][0][k]+rate*y[hl][k]*note[hl][0]

        accurate=accuraterate(w,hl,hp) #計算現在鍵結值的正確率
        # print(f"accurate:{accurate}")
        
        if accurate>maxaccurate: #正確率大於最大的正確率 
            answ=[] #紀錄鍵結值
            for j in range(hl+1):
                tmp1=[]
                for k in range(len(w[j])):
                    tmp2=[]
                    for l in range(len(w[j][k])):
                        tmp2.append(w[j][k][l])
                    tmp1.append(tmp2)
                answ.append(tmp1)

            maxaccurate=accurate


    
    #======================結果=========================================
    plt.subplot(2, 2, 1)
    for i in range(len(listxy)):
        if(listexp[i] == 1):
            plt.scatter(listxy[i][0],listxy[i][1],s=5,color = 'cyan')
        elif(listexp[i]==0):
            plt.scatter(listxy[i][0],listxy[i][1],s=5,color = 'gray')
    plt.title("beforetrain")

    cnt=0

    plt.subplot(2, 2, 2)
    for i in range(len(listxy)):
        tmp=[-1]  # 輸入
        for j in range(2):
            tmp.append(listxy[i][j])
        
        for j in range(hl): #層
            tmp2=[-1]
            for k in range(hp): #神經元
                sum=0
                for l in range(len(tmp)):
                    sum+=tmp[l]*answ[j][k][l] #內積
                tmp2.append(activefunction(sum))
            tmp=tmp2 #算出來的結果變成新的輸入
        
        sum=0
        for l in range(len(tmp)): # 輸出層
            sum+=answ[hl][0][l]*tmp[l]
        sum=activefunction(sum)

        if sum<0.5:
            plt.scatter(listxy[i][0],listxy[i][1],s=5,color = 'gray')
            if listexp[i]==0:
                cnt+=1
        if sum>=0.5:
            plt.scatter(listxy[i][0],listxy[i][1],s=5,color = 'cyan')
            if listexp[i]==1:
                cnt+=1
    plt.title("aftertrain")
    print(cnt/len(listexp))
    print(maxaccurate)
    plt.show()



def accuraterate(wchan,hl,hp):
    global maxaccurate,listexp,listxy

    E=0 #誤差
    Eav=0 #平均誤差

    # print(f"w={w}")
    accurate=0
    
    for i in range(len(listxy)):
        tmp=[-1]  # 輸入
        for j in range(2):
            tmp.append(listxy[i][j])
        
        for j in range(hl): #層
            tmp2=[-1]
            for k in range(hp): #神經元
                sum=0
                for l in range(len(tmp)):
                    sum+=tmp[l]*wchan[j][k][l] #內積
                tmp2.append(activefunction(sum))
            tmp=tmp2 #算出來的結果變成新的輸入
        
        sum=0
        for l in range(len(tmp)): # 輸出層
            sum+=wchan[hl][0][l]*tmp[l]
        sum=activefunction(sum)

        if sum<0.5 and listexp[i]==0: #正確率
            accurate+=1
        if sum>=0.5 and listexp[i]==1:
            accurate+=1

        E+=(listexp[i]-sum)*(listexp[i]-sum) #誤差
    
    
    # E/=2
    # Eav=E/len(listxy) 
    accurate/=len(listxy)
    return accurate
    # print(f"平均誤差為:{Eav}")
    # print(f"正確率:{accurate}")

        
        

learnnumbertext=tk.StringVar() #學習次數
learnnumber=tk.Entry(window,textvariable=learnnumbertext)
learnnumber.place(x=60,y=60)
learnnumberlb=tk.Label(window,font="微軟正黑體 8 bold",text="學習次數")
learnnumberlb.place(x=10,y=60)

learingtext=tk.StringVar() #學習率
learingentry=tk.Entry(window,textvariable=learingtext)
learingentry.place(x=60,y=10)
learninglb=tk.Label(window,font="微軟正黑體 8 bold",text="學習率")
learninglb.place(x=10,y=10)

hiddenlayertext=tk.StringVar() #隱藏層數量
hiddenlayernumber=tk.Entry(window,textvariable=hiddenlayertext)
hiddenlayernumber.place(x=80,y=110)
hiddenlayerlb=tk.Label(window,font="微軟正黑體 8 bold",text="隱藏層數量")
hiddenlayerlb.place(x=10,y=110)

hiddenpointtext=tk.StringVar() #隱藏層神經元
hiddenpointnumber=tk.Entry(window,textvariable=hiddenpointtext)
hiddenpointnumber.place(x=80,y=160)
hiddenpointlb=tk.Label(window,font="微軟正黑體 8 bold",text="隱藏層神經元")
hiddenpointlb.place(x=10,y=160)

selectfilebt=tk.Button(window,text="選擇檔案",command=browse_file)
selectfilebt.place(x=300,y=50)

filename=tk.StringVar()
filelb=tk.Label(window,font="微軟正黑體 8 bold",textvariable=filename)
filelb.place(x=250,y=80)

submitbt=tk.Button(window,text="送出",command=submit)
submitbt.place(x=160,y=200)

window.geometry('750x250')
window.title("project2")
window.mainloop()