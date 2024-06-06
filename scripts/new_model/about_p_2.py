import numpy as np
import matplotlib.pyplot as plt

p=np.linspace(0.1,1,10)
w = 0.5
beta = 0.5
gamma_inverse= 7
gamma = 1/gamma_inverse
L=0.2
K=0.2
R0 = (beta/(K*gamma))*(p*K+(1-p)*w*(1-L+K*L))
plt.title('Under the negative circumstance')
plt.xlabel('Unvaccinated rate')
plt.ylabel('Basic regeneration cofficient R0')

plt.bar(p,R0,width=0.05,color='coral')
plt.show()


L=0.5
K=0.8
R0 = (beta/(K*gamma))*(p*K+(1-p)*w*(1-L+K*L))
plt.title('Under the positive circumstance')
plt.xlabel('Unvaccinated rate')
plt.ylabel('Basic regeneration cofficient R0')

plt.bar(p,R0,width=0.05)
plt.show()