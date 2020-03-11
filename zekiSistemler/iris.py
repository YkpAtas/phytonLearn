import matplotlib.pyplot as plt # görselleştirme için 
import pandas as pd# veri seti işlemleri için
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



#sklearn kütüphanesi makine öğrenmenim algoraitmalarının uygulama ve sonuçları
#değerlendirmek için kullanılmışrır

# Veri setinin yüklenmesi

#iris.csv de bulunan verileri pandas alır ve okur
iris_dataset = pd.read_csv('/home/yakup/Desktop/phyton/zekiSistemler/iris.csv')


# Bağımlı ve bağımsız değişkenlerin oluşturulması
X = iris_dataset.values[:, 0:4]# 0 dan 3 kadar olan karakterler yani sayilar bağımsız değişkenlerdir
Y = iris_dataset.values[:, 4] # 4. bağımlı değişkenlerdir.


# Veri kümesinin eğitim ve test verileri olarak ayrılması
#x_train = bağımsız değişkenler arasında rastgele seçilen %20 lik kısımlık veri olan eğitim verisidir.
#x_test= x_train karşılaştırma yapılacağı 
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20, random_state=7)


print('1: veri setlerini görmek için\n2: veri setinin istatistik gösterimi\n3; veri setinin türlerine göre dağılımı')
print('4: veri setinin 3 adımdaki istatistiksel verilere göre grafik dağılımı')
print('5: Algoritmaların uygulanması ve sonuçlarının değerlendirilmesi')
print('6: Uygun algoritmanın seçilmesi ve tahmin yapılması')
sayi =int(input('lütfen işlem giriniz'))

if(sayi==1):
    print('X-train veri seti random-state=7 şeklinde sıralanmıştır')
    print(X_train) 
    print('X-train veri seti random-state=7 şeklinde sıralanmıştır')
    print(X_test) 
        

if(sayi==2):
    print('istatistik gösterimi')
    print(iris_dataset.describe())
        

if(sayi==3):  
    print("verilerin tür değişkenine göre dağılımı")
    print(iris_dataset.groupby('variety').size())
        
if(sayi==4): 
    print('kutu grafigi') 
    iris_dataset.plot(kind='box', subplots=True, sharex=False, sharey=False)
    plt.show()
       

    print('histogram')  
    iris_dataset.hist()
    plt.show()


    print('scatter plot matrix')
    scatter_matrix(iris_dataset)
    plt.show()
        
if(sayi==5):
    print('bu adımda sklearn kütüphanesindeki modelleri deniyoruz\n sonuçları karşılaştırıp en doğru sonuç veren model seçilecektir.')
    print('Cross validation: farklı modeller deneme sürecidir. modeller farklı standart sapma istatistikleri vericektir. en düşük olanı seçilecektir')

    #'Modellerin listesinin olusturulmasi
    models = [
        ('LR', LogisticRegression()),
        ('LDA', LinearDiscriminantAnalysis()),
        ('KNN', KNeighborsClassifier()),
        ('DT', DecisionTreeClassifier()),
        ('NB', GaussianNB()),
        ('SVM', SVC())
    ]
    # Modeller için 'cross validation' sonuçlarının  yazdırılması
    results = []
    names = []
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=7)
        cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring="accuracy")
        results.append(cv_results)
        names.append(name)
        print("%s: %f (%f)" % (name, cv_results.mean(), cv_results.std()))

if(sayi==6):
    print('------------------------------------------------------------------------')
    print('standart sapma değerine göre en düşük ss değeri veren model seçilmiştir..')
    svc = SVC()# en uygun model svc seçilmiştir
    svc.fit(X_train, Y_train) # fit bir eğitim fnksidir. verinin sadece %20si sayılar arasında random seçilerek göderiliyor.bu fonksiyona X_train bağımsız değişkenleri ile y_train bağımlı değişkeni öğrenmeye çalışıyor.
    predictions = svc.predict(X_test) #x_test verileri ile tahmin yapabilmek için veriler svc.predict() fnksiyonuna gönderilip sonuçlar predictions ta tutuluyor.
    print('NOT :burda cross_validation yontemi kullanılmıştır. bu yontem eğitim verilerini seçerken farklı bir yöntem dener. bu yüzden farklı sönuçlar ortaya çıkabilir.')
    print('NOT:cross_validation yöntemi:örneğin 10 eğitim veriniz bulunmaktadır. yöntem bu on veriden 1 tanesini test verisi 9 unu da eğitim olarak seçer. 10 seçilen eğitim veri kadar tekrar eder.\nrandom olarak ilerler bu nedenle bazı modeller tek eğitim ve test verilerinde doğru sonuçlar verirken. cross_validation yönteminde kötü sonuçlar verebilir')
    print('------------------------------------------------------------------------')
    print('accuracy degeri :', accuracy_score(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))
    print(classification_report(Y_test, predictions))

    print('-------------------------------')
    print('yanlış Eşleştirmeler:')
    for i in range(0,len(X_test)):
        if(Y_test[i]!=predictions[i]):
            print(Y_test[i],predictions[i])
        