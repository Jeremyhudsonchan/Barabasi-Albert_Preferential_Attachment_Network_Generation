import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

def algo_2(A, Beta, Gamma, delta_t, j):
    S = np.ones([N,200])
    I = np.zeros([N,200])
    R = np.zeros([N,200])
    S[j, 0] = 0
    I[j, 0] = 1
    for t in range(200 - 1):
        S[:,t+1] = S[:,t]
        I[:,t+1] = I[:,t]
        R[:,t+1] = R[:,t]
        for i in range(N):
            if S[i, t] == 1:
                n = 0
                for j in range(N):
                    n=n+A[i,j]*I[j,t]
                if np.random.rand() < 1 - np.exp(-delta_t * n * Beta):
                    I[i, t+1] = 1
                    S[i, t+1] = 0
            if I[i, t] == 1:
                if np.random.rand() < 1 - np.exp(-delta_t * Gamma):
                    R[i, t+1] = 1
                    I[i, t+1] = 0
    return S, I, R

# Read in Files
ba = nx.read_adjlist('/Users/jeremyhudsonchan/Dropbox/Files/Boston_College_Courses/Semester_7/Csci3391/Boston_College_CSCI3391/Homeworks/homework_2/homework_2b/ba_graph.adjlist')
ws = nx.read_adjlist('/Users/jeremyhudsonchan/Dropbox/Files/Boston_College_Courses/Semester_7/Csci3391/Boston_College_CSCI3391/Homeworks/homework_2/homework_2b/ws_graph.adjlist')

# Draw Graph
ba_graph = nx.draw_spring(ba)
plt.show()
ws_graph = nx.draw_spring(ws)
plt.show()

print("ba_graph assortativity coefficient: ", nx.degree_assortativity_coefficient(ba))
print("ws_graph assortativity coefficient: ", nx.degree_assortativity_coefficient(ws))

# Parameters
N = 100
Beta = 1
Gamma = 1
delta_t = 0.1
T = 200

ba_adj = nx.adjacency_matrix(ba).todense()
ba_den = nx.density(ba)

ws_adj = nx.adjacency_matrix(ws).todense()
ws_den = nx.density(ws)

for times in range(100):
    ba_S , ba_I, ba_r = algo_2(ba_adj, Beta, Gamma, delta_t, np.random.choice(np.arange(ba_adj.shape[0])))
    ba_I_sum = np.sum(ba_I, axis=0)
    plt.plot(ba_I_sum)
plt.title('ba_graph')
plt.show()

for times in range(100):
    ws_S, ws_I, ws_r = algo_2(ws_adj, Beta, Gamma, delta_t, np.random.choice(np.arange(ws_adj.shape[0])))
    ws_I_sum = np.sum(ws_I, axis=0)
    plt.plot(ws_I_sum, label = 'ws')

plt.title('ws_graph')
plt.show()
