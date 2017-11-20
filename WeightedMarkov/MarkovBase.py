import random
import itertools as it
import cvxopt
from cvxopt import matrix, solvers
from fractions import Fraction
from copy import deepcopy
from collections import defaultdict
from numpy import unique
import numpy as np
import re;

class MarkovBase:
    def __init__(self, X=[], nStates=0, order= 1):
        assert order   > 0, 'Markov Chain order must be > 0'
        assert nStates > 0, 'Markov Chain number of States must be > 0'
        self.X = X
        self.nStates = nStates
        self.order = order
        self.states= {}
    
    @staticmethod
    def Score(a, b, printOut=True, msg=None):
        n=0;
        t=0;
        z = zip(a,b)
        correctClass=defaultdict(int)
        totalClass=defaultdict(int)

        for c in z:
            totalClass[c[0]] += 1;

            if(c[0] == c[1]):
                correctClass[c[0]] += 1;
                t+= 1
            n += 1

        if (printOut):
            print("=======================Metrics : ", msg)
            print("orig=>{}\npred=>{}".format(a[0:80], b[0:80]))
            print("Total %d, correct %d, acc: %3.2f"%(n,t,t/n))
            for i,c in totalClass.items():
                acc = correctClass[i]/c
                print("class:{} total:{}, correct:{}, accuracy:{}".format(i, c, correctClass[i], acc))

        return n, t, totalClass
    
    # Must provide classes encoded by 0, 1, 2 etc
    def Freq(self, s1, s2 = None):
        if len(s1) <= 0:  return None;
        if (s2 is None):  s2 = s1[1:]

        F=np.zeros((self.nStates, self.nStates))
        for z in zip(s2,s1):
            F[(z)] += 1

        F=np.matrix(F)
        div = F.sum(axis=0)
        for o,c in enumerate(div.flat):
            if (c==0):
                F[:,o]=1
        
        P=F.copy();
        P = P/P.sum(axis=0)
        return F, P

    def M(self,m, name="", useFrac=False, call_display=True, showdim=True, precision=4):
        np.set_printoptions(precision=precision, linewidth=180)
        name = name + " =" if name != "" else ""
        dim = "";
        if (showdim):
            dim = " \\times ".join(map(str, (m.shape) )) ;
        if (useFrac):
            m=np.array([ str(Fraction(_).limit_denominator()) for _ in m.flat]).reshape(m.shape)
        s = str(m).replace("'", '')
        s=s.replace('\\\\', '\\')
        s = s.replace('[', '')
        s = s.replace(']', '')
        s = s.replace('\n', '\\\\\\\\<NEW-LINE>')
        s = re.sub( '\s+', ' ', s ).strip()
        s = s.replace('<NEW-LINE>', "\n")
        s = re.sub('\n\s+', '', s)
        s = s.replace(' ', ' & ')
        s = name + "\\begin{bmatrix}\n" + s + "\n\\end{bmatrix}" +  dim + "\n"
        #print self.a
        if ( call_display):
            display(Math(s))
        return s;    

    
    #---- Private stuff
    def Encode(self, y):
        l = preprocessing.LabelEncoder()
        y = l.fit_transform(y);
        return y, l.classes_


    
