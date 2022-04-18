import random
import numpy as np

def data_gen(s1, s2, p, q, r, N):
    data = []
    for i in range(N):
        coin = random.random()
        if 0 <= coin < s1:
            side = np.random.binomial(1,p)
        elif s1 <= coin < s1 + s2:
            side = np.random.binomial(1,q)
        else:
            side = np.random.binomial(1,r)
        data.append(side)
    return data

  
def EM(theta, e, y, max_epoch):
    s1 = theta[0]
    s2 = theta[1]
    p  = theta[2]
    q  = theta[3]
    r  = theta[4]
    N  = len(y)
    i = 0
    threshold = 1 
    while(i < max_epoch and threshold >= e):
        # Expectation
        a = np.zeros(N) 
        b = np.zeros(N)
        for j in range(N):
            a[j] = (s1*pow(p,y[j])*pow(1-p,1-y[j]))/(s1*pow(p,y[j])*pow(1-p,1-y[j])+s2*pow(q,y[j])*pow(1-q,1-y[j])+(1-s1-s2)*pow(r,y[j])*pow(1-r,1-y[j]))
            b[j] = (s2*pow(q,y[j])*pow(1-q,1-y[j]))/(s1*pow(p,y[j])*pow(1-p,1-y[j])+s2*pow(q,y[j])*pow(1-q,1-y[j])+(1-s1-s2)*pow(r,y[j])*pow(1-r,1-y[j]))           
        # Maximization
        s1_next = 1/N * sum(a)
        s2_next = 1/N * sum(b)
        p_next = sum([a[j]*y[j] for j in range(N)]) / sum(a)
        q_next = sum([b[j]*y[j] for j in range(N)]) / sum(b)
        r_next = sum([(1-a[j]-b[j])*y[j] for j in range(N)]) / sum([(1-a[j]-b[j]) for j in range(N)])
        # Threshold 
        threshold = np.linalg.norm(np.array([s1-s1_next,s2-s2_next,p-p_next,q-q_next,r-r_next]),ord = 2)
        s1 = s1_next
        s2 = s2_next
        p  = p_next
        q  = q_next
        r  = r_next
        i += 1
        print(i,[s1,s2,p,q,r])
    return s1,s2,p,q,r       

if __name__ == "__main__":
    N = 10
    max_epoch = 10
    y = data_gen(0.2,0.3,0.1,0.9,0.5,N)
    theta0 = [0.4,0.5,0.2,0.6,0.8]
    res = EM(theta0,10e-20,y,max_epoch)
    # print(res)