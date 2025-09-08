import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

rng = np.random.default_rng(2)

def make_data(n=800, d=8):
    X1 = rng.normal(0,1,size=(n//2,d))
    X2 = rng.normal(0.7,1,size=(n//2,d))
    X = np.vstack([X1,X2]); y = np.array(*(n//2)+[11]*(n//2))
    return X,y

def q_like_kernel(a,b,scale=1.3):
    fa = np.concatenate([np.cos(scale*a), np.sin(scale*a)])
    fb = np.concatenate([np.cos(scale*b), np.sin(scale*b)])
    fa /= np.linalg.norm(fa); fb /= np.linalg.norm(fb)
    return float(fa@fb)

def build_K(XA,XB):
    K = np.zeros((len(XA),len(XB)))
    for i in range(len(XA)):
        for j in range(len(XB)):
            K[i,j] = q_like_kernel(XA[i], XB[j])
    return K

if __name__ == "__main__":
    X,y = make_data()
    Xtr,Xte,ytr,yte = train_test_split(X,y,test_size=0.3,random_state=0,stratify=y)
    # Quantum-like kernel SVM
    Ktr = build_K(Xtr,Xtr); Kte = build_K(Xte,Xtr)
    svm_q = SVC(kernel="precomputed", C=1.0).fit(Ktr,ytr)
    acc_q = accuracy_score(yte, svm_q.predict(Kte))
    # RBF baseline
    svm_r = SVC(kernel="rbf", C=1.0, gamma="scale").fit(Xtr,ytr)
    acc_r = accuracy_score(yte, svm_r.predict(Xte))
    print({"acc_quantum_like": round(acc_q,4), "acc_rbf": round(acc_r,4)})
