import os
import xlsxwriter




FOLDER_PATH=r'C:\\Users\\yakup.atas\\Desktop\\yakup\\yakup\\veriler\\tako yedekler'

def findLineName(dir):
    takoIndirilenlerHamVeri=[]
    fileNames = os.listdir(dir)

    for fileName in fileNames:
        takoIndirilenlerHamVeri.append(fileName)
    cutPlaka(takoIndirilenlerHamVeri)

def cutPlaka(deneme):
    plakalar=[]
    soforler=[]
    tarihPlaka=[]
    tarihSofor=[]
    for name in deneme:
        if(name[0:1]=='C'):
            soforler.append(name[16:24])
            tarihSofor.append(name[2:10])
        elif(name[0:1]=='M'):
            plakalar.append(name[16:24])
            tarihPlaka.append(name[2:10])
        
    #ekranaBas(plakalar)
    excelAktar(plakalar,soforler,tarihPlaka)



def excelAktar(plaka,sofor,tarih):
    workbook=xlsxwriter.Workbook("C:/Users/yakup.atas/Desktop/Verileri indirilmiş araçlar.xlsx")
    worksheet=workbook.add_worksheet()
    bold=workbook.add_format({'bold':True})
    worksheet.write('A1', 'PLAKALAR', bold)
    worksheet.write('B1', 'TARİH', bold)

  
    row=0
    col=0
   

    for i in range(len(plaka)):
        worksheet.write(row+1,col, plaka[i])
        worksheet.write(row+1,col+1, tarih[i])
        row+=1
    workbook.close()
    print("yapıldı")
"""
       try:
            worksheet.write(row,col+2,sofor[i])
        except:
            worksheet.write(row,col+1,"NULL")
       
       """


def ekranaBas(liste):
    for i in liste:
        print(i)        



if __name__== '__main__':
    findLineName(FOLDER_PATH)

