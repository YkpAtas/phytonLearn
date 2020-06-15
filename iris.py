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
from sklearn import metrics
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier,VotingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier

#from sklearn.preprocessing import StandartScaler



#sklearn kütüphanesi makine öğrenmenim algoraitmalarının uygulama ve sonuçları
#değerlendirmek için kullanılmışrır

# Veri setinin yüklenmesi

#iris.csv de bulunan verileri pandas alır ve okur
iris_dataset = pd.read_csv('iris.csv')


# Bağımlı ve bağımsız değişkenlerin oluşturulması
X = iris_dataset.values[:, 0:4]# 0 dan 3 kadar olan karakterler yani sayilar bağımsız değişkenlerdir
Y = iris_dataset.values[:, 4] # 4. bağımlı değişkenlerdir.


# Veri kümesinin eğitim ve test verileri olarak ayrılması
#x_train = bağımsız değişkenler arasında rastgele seçilen %20 lik kısımlık veri olan eğitim verisidir.
#x_test= x_train karşılaştırma yapılacağı 
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.20,random_state=7)


print('1: veri setlerini görmek için\n2: veri setinin istatistik gösterimi\n3; veri setinin türlerine göre dağılımı')
print('4: veri setinin 3 adımdaki istatistiksel verilere göre grafik dağılımı')
print('5: Algoritmaların uygulanması ve sonuçlarının değerlendirilmesi')
print('6: Uygun algoritmanın seçilmesi ve tahmin yapılması')
print("7: Basit modeller ve tahmin sonuçları")
print("8: Karmaşık modeller ve tahmin sonuçları")
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
    
    print(classification_report(Y_test, predictions))
    print(confusion_matrix(Y_test, predictions))
    print('-------------------------------')
    print('yanlış Eşleştirmeler:')
    for i in range(0,len(X_test)):
        if(Y_test[i]!=predictions[i]):
            print(Y_test[i],predictions[i])

if(sayi==7):

    print("Basit Modellemeler\nSupport Vector Classification(1)\nDecision Tree Classifier(2)")
    print("\nKNeighborsClassifier(3)")
    model=int(input("İşlem girinizz::"))
    if(model==1):
        print("\nkernal= linear olduğununda\n----------------------------")
        svcclassifier=SVC(kernel='linear')
        svcclassifier.fit(X_train,Y_train)
        y_pred=svcclassifier.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        #print(y_pred)
        print("\nkernal= poly olduğununda\n----------------------------")
        svcclassifier=SVC(kernel='poly')
        svcclassifier.fit(X_train,Y_train)
        y_pred=svcclassifier.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        print("\nkernal= rbf olduğununda\n----------------------------")
        svcclassifier=SVC(kernel='rbf')
        svcclassifier.fit(X_train,Y_train)
        y_pred=svcclassifier.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])

    if(model==2):
        print("\n------------------\n")
        print("Decision Tree Classifier Default Değerler\n--------------------")
        dtc=DecisionTreeClassifier()
        dtc.fit(X_train,Y_train)
        y_pred=dtc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        print("\n------------------\n")
        print("Decision Tree Classifier Criterion=Gini Random_state=0 Değerler\n--------------------")
        dtc=DecisionTreeClassifier(splitter='best',criterion='gini')
        dtc.fit(X_train,Y_train)
        y_pred=dtc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        
        print("\n------------------\n")
        print("Decision Tree Classifier Criterion=Entropy Random_state=7 Değerler\n--------------------")
        dtc=DecisionTreeClassifier(random_state=7,criterion='entropy')
        dtc.fit(X_train,Y_train)
        y_pred=dtc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
    if(model==3):
        print("\n------------------\n")
        print("KNeinghbors Classifier default Değerler\n--------------------")
        knc=KNeighborsClassifier()
        knc.fit(X_train,Y_train)
        y_pred=knc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        
        print("\n------------------\n")
        print("KNeinghbors Classifier algorithm="'ball_tree'",weights="'uniform'",n_neighbors=30 Değerler\n--------------------")
        knc=KNeighborsClassifier(n_neighbors=30,algorithm='ball_tree', weights="uniform")
        knc.fit(X_train,Y_train)
        y_pred=knc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        
        print("\n------------------\n")
        print("KNeinghbors Classifier algorithm='kd_tree', weights="'uniform'",n_neighbors=23 Değerler\n--------------------")
        knc=KNeighborsClassifier(algorithm='kd_tree', weights="uniform",n_neighbors=23)
        knc.fit(X_train,Y_train)
        y_pred=knc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])

if(sayi==8):
    print("Ensamble(kollektif) Modellemeler\nRandom Forest Classifier(1)\nAda Boost Classifier(2)")
    print("Bagging clasifier(3)")
    print("Voting Classifer (4)")
    model=int(input("İşlem girinizz::"))
    if(model==1):
        print("Random Forest Classifier Default Değerler\n--------------------")
        rfc=RandomForestClassifier(n_estimators=100)
        rfc.fit(X_train,Y_train)
        y_pred=rfc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
    if(model==2):
        print("Ada Boost Classifier Default Değerler\n--------------------")
        abc=AdaBoostClassifier(n_estimators=10,learning_rate=1)
        abc.fit(X_train,Y_train)
        y_pred=abc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        print("Using Different Base Learners")
        svc=SVC(probability=True,kernel='linear')
        abc =AdaBoostClassifier(n_estimators=50, base_estimator=svc,learning_rate=1)
        abc.fit(X_train,Y_train)
        y_pred=abc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        
    if(model==3):
        print("Bagging Classifier Default Değerler\n--------------------")
        bc=BaggingClassifier(n_estimators=10)
        bc.fit(X_train,Y_train)
        y_pred=bc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])
        print("Using Different Base Learners")
        svc=SVC(probability=True,kernel='linear')
        bc =BaggingClassifier(n_estimators=50, base_estimator=svc)
        bc.fit(X_train,Y_train)
        y_pred=bc.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])   
    
    if(model==4):
        print("Voting Classifer")
        vc1=SVC(kernel='linear')
        vc2=DecisionTreeClassifier(splitter='best',criterion='gini')
        vc3=KNeighborsClassifier(n_neighbors=30,algorithm='ball_tree', weights="uniform")
        vc1=VotingClassifier(estimators=[('lr',vc1),('rf',vc2),('gnb',vc3)],voting='hard')
        vc1.fit(X_train,Y_train)
        y_pred=vc1.predict(X_test)
        print("\nconfusion_matrisi hesabı\n")
        print(confusion_matrix(Y_test, y_pred))
        print("\nconfusion_raporu\n")
        print(classification_report(Y_test, y_pred))
        print('accuracy degeri :', metrics.accuracy_score(Y_test, y_pred))
        print('-------------------------------')
        print('yanlış Eşleştirmeler:')
        for i in range(0,len(X_test)):
            if(Y_test[i]!=y_pred[i]):
                print(Y_test[i],y_pred[i])   
 
        
