# -*- coding: utf-8 -*-
import numpy as np
data = np.loadtxt("./data/data.csv")
X = data[:,:-2]
Y = data[:,-2:]
Yheat = Y[:,0]
Ycool = Y[:,1]

import random

nb_exemples = 768
nb_attributs = 8
nb_tot_attributs = 10

# Partie 1 (10 points)

# Question 1 (4 points)

# Réservez 25\% des exemples pour le test et 75\% pour l'apprentissage : 1 point
from sklearn.cross_validation import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Yheat,test_size = 0.25, random_state=random.seed())

# Apprenez une fonction de régression linéaire $f_{heat}$ permettant de prédire les charges de chauffage en fonction des attributs descriptifs : 1 point

from sklearn.linear_model import LinearRegression

lr = LinearRegression()

lr.fit(X_train, Y_train)

# Affichez le score : 0.5 pt pour la commande, 0.5 pour le résultat
print("Score de la régression : ", lr.score(X_test, Y_test))

# Expliquez précisément ce que représente ce score.

## le R2 score est décrit à la page suivante : http://scikit-learn.org/stable/modules/model_evaluation.html#r2-score

## Score : 1 - \sum (y_i - \hat{y}_i)^2 /\sum (y_i - mean(y_i))^2

# Question 2 (3.5 points)

#  \'Ecrivez le code permettant de trouver l'attribut à éliminer : 3 points 
from sklearn.cross_validation import KFold

kf=KFold(len(X_train),n_folds=10,shuffle=True)

scores = []

for i in range(nb_attributs):
    score = 0
    Xnew = np.delete(X_train, i, axis=1)
    for learn,test in kf:
        X_train_val=[Xnew[j] for j in learn]
        Y_train_val=[Y_train[j] for j in learn]
        lr.fit(X_train_val, Y_train_val)
        X_test_val=[Xnew[j] for j in test]
        Y_test_val=[Y_train[j] for j in test]
        score += lr.score(X_test_val,Y_test_val)
    scores.append(score)

AttOpt = scores.index(max(scores))

# Quel est cet attribut ?  0.5 point 
print("Numero d'attribut à enlever : ", AttOpt)

# Question 3 (2.5 points)

# Est-ce une bonne idée d'éliminer un attribut ? Comparez les scores obtenus avec élimination et sans élimination. 2 points

Xnew = np.delete(X_train, (AttOpt), axis=1)
Xtestnew = np.delete(X_test, (AttOpt), axis=1)

lr.fit(Xnew, Y_train)


print("Score de la régression après suppression d'un attribut : ", lr.score(Xtestnew, Y_test))
# Commentaire : 0.5 point
# Amélioration à peine perceptible.


# Partie 2 (10 points)




target = np.loadtxt('./labels', dtype='int')

print(len(Y[target==0]), Y[target==0].sum()/len(Y[target==0]))
print(len(Y[target==1]), Y[target==1].sum()/len(Y[target==1]))
print(len(Y[target==2]), Y[target==2].sum()/len(Y[target==2]))

## Question 1 : 5 points

# séparer les données de travail en des données d'apprentissage et de test : 1 point

X_train, X_test, Y_train, Y_test = train_test_split(X,target,test_size = 0.25, random_state=random.seed())

# sélectionner les hyperparamètres des algorithmes d'apprentissage sur l'échantillon d'apprentissage par validation croisée : 2 points

kf=KFold(len(X_train),n_folds=10,shuffle=True)

# k-plus proches voisins
from sklearn import neighbors

scores = []

for k in range(1, 11):
    clf = neighbors.KNeighborsClassifier(k)
    score = 0
    for learn,test in kf:
        X_train_val=[X_train[j] for j in learn]
        Y_train_val=[Y_train[j] for j in learn]
        clf.fit(X_train_val, Y_train_val)
        X_test_val=[X_train[j] for j in test]
        Y_test_val=[Y_train[j] for j in test]
        score += clf.score(X_test_val,Y_test_val)
    scores.append(score)

k_opt = scores.index(max(scores)) + 1
    
print("nombre de voisins optimal : ",k_opt)

# SVM à noyau RBF
from sklearn.svm import SVC

scores = []

for k in range(1, 11):
    clf = SVC(gamma= k*0.1, kernel = 'rbf')
    score = 0
    for learn,test in kf:
        X_train_val=[X_train[j] for j in learn]
        Y_train_val=[Y_train[j] for j in learn]
        clf.fit(X_train_val, Y_train_val)
        X_test_val=[X_train[j] for j in test]
        Y_test_val=[Y_train[j] for j in test]
        score += clf.score(X_test_val,Y_test_val)
    scores.append(score)

gamma_opt = (scores.index(max(scores)) + 1) * 0.1
    
print("gamma optimal : ",gamma_opt)


# SVM à noyau polynomial
## from sklearn.svm import SVC

## scores = []

## for k in range(1, 2):
##     clf = SVC(kernel = 'poly')
##     score = 0
##     for learn,test in kf:
##         X_train_val=[X_train[j] for j in learn]
##         Y_train_val=[Y_train[j] for j in learn]
##         clf.fit(X_train_val, Y_train_val)
##         X_test_val=[X_train[j] for j in test]
##         Y_test_val=[Y_train[j] for j in test]
##         score += clf.score(X_test_val,Y_test_val)
##     scores.append(score)

## degre_opt = (scores.index(max(scores)) + 1) 
    
## print("degre optimal : ",degre_opt)


# afficher les scores de ces algorithmes évalués sur l'échantillon test : 2 points 
    
clf = neighbors.KNeighborsClassifier(k_opt)
clf.fit(X_train, Y_train)
sk = clf.score(X_test, Y_test)
Y_k_pred = clf.predict(X_test)
print("Score des k plus proches voisins, avec k = ",k_opt," : ", sk)

from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, Y_train)
st = clf.score(X_test, Y_test)
Y_AD_pred = clf.predict(X_test)
print("Score des arbres de décision (paramètres par défaut) : ", st)

from sklearn.svm import SVC
clf = SVC(kernel = 'linear')
clf.fit(X_train, Y_train)
sl = clf.score(X_test, Y_test)
print("Score du SVM à noyau linéaire : ", sl)


clf = SVC(gamma= gamma_opt, kernel = 'rbf')
clf.fit(X_train, Y_train)
sr = clf.score(X_test, Y_test)
print("Score du SVM à noyau rbf : ", sr)


# Question 2 : intervalles de confiance : 2 points 
import math

N = len(X_test)
print("Intervalles de confiance à 95%")

rk = 1.96*math.sqrt(sk*(1-sk)/N)
print("k plus proches voisins : ]",1-sk-rk,1-sk+rk,"[")
rt = 1.96*math.sqrt(st*(1-st)/N)
print("arbres de décision : ]",1-st-rt,1-st+rt,"[")
rl = 1.96*math.sqrt(sl*(1-sl)/N)
print("svm linéaire : ]",1-sl-rl,1-sl+rl,"[")
rr = 1.96*math.sqrt(sr*(1-sr)/N)
print("svm rbf : ]",1-sr-rr,1-sr+rr,"[")

# Question 3 (3 points)

# Commentez les résultats obtenus.

# AD > k-ppv > SVM RBF > SVM Lin : 1 point (ou autre réponse compatible avec les résultats donnés)

#Quel algorithme recommanderiez  vous d'utiliser ? Avec quelles garanties de résultat ?
# AD est souvent bien meilleur que kppv mais les intervalles de confiance se chevauchent. En revanche, le plus souvent, les intervalles de confiance de AD et des SVM sont disjoints. Recommander AD sans garantie que le résultat soit meilleur que k-ppv, à moins d'affiner les tests : 2 points

# BONUS

# Les deux meilleurs algorithmes sont-ils départageables avec le test de McNemar ? : 2 points


n00 = n01 = n10 = n11 = 0
for i in range(len(X_test)):
    if Y_AD_pred[i] == Y_test[i]:
        if Y_k_pred[i] == Y_test[i]:
            n11 += 1
        else:
            n10 += 1
    elif Y_k_pred[i] == Y_test[i]:
            n01 += 1
    else:
            n00 += 1

x = (abs(n01-n10)-1)**2/(n01+n10)

print("Critère de McNemar ", x)

if (x > 3.841459):
        if n10 > n01:
                print("le classifieur appris par arbre de décision est significativement meilleur que le classifieur appris par les k-ppv")
        else:
                print("le classifieur appris par les k-ppv est significativement meilleur que le classifieur appris par arbre de décision")
else:
        print("le test ne permet pas de départager les 2 classifieurs")
