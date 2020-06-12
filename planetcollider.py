# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib import animation
from scipy.integrate import solve_ivp
from timeit import default_timer as timed
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import axes3d

G = 6.674e-11
particles = 2400
planet = np.loadtxt(f'./Planet{particles}.dat')
r = np.amax(planet[:,0]) - np.amin(planet[:,0])
planet[:,3] = - planet[:,1] * 0.0004
planet[:,4] = planet[:,0] * 0.0004
planet2 = np.copy(planet)
planet2[:,1] += 1.5*r
planet2[:,0] += - 1*r
planet2[:,4] += - 2e4
data = np.vstack((planet, planet2))

total = len(data)
energy = data[:,8] / data[:,7] / 0.4
data = np.append(data, energy.reshape(total, -1), axis=1)
len1, len2 = len(planet), len(planet2)

combinations = np.array(list(product(np.arange(total), np.arange(total))))
i, j = combinations[:,0], combinations[:,1]

def FORCE(t, data):
    data = data.reshape(total, -1)    
    position = data[:,0:3]
    velocity = data[:,3:6]
    mass = np.mean(data[:,6])
    density = data[:,7].reshape(total, -1)
    pressure = data[:,8].reshape(total, -1)
    energy = data[:,9].reshape(total, -1)    
    h = np.full((total**2, 1),1.3e7)
    dx = position[i] - position[j]
    distance = cdist(position, position).reshape(total**2, -1)
    R = distance / h
    alpha = 3 / (2 * np.pi * h**3)    
    h[h==0]=1e-9
    distance[distance==0]=1e-9
    R[R==0]=1e-9
    
    r1 = R < 1
    r2 = np.logical_and(R >= 1, R < 2)
    r3 = R >= 2
    
    w = np.zeros((total**2, 1))
    w[r1] = alpha[r1] * (2/3 - R[r1]**2 + 0.5 * R[r1]**3)
    w[r2] = alpha[r2] * 1/6 * (2 - R[r2])**3
    
    dwdx = np.zeros((total**2, 3))
    dx0 = dx[:,0].reshape(total**2, -1)
    dx1 = dx[:,1].reshape(total**2, -1)
    dx2 = dx[:,2].reshape(total**2, -1)
    dwdx0 = dwdx[:,0].reshape(total**2, -1)
    dwdx1 = dwdx[:,1].reshape(total**2, -1)
    dwdx2 = dwdx[:,2].reshape(total**2, -1)
    dwdx0[r1] = alpha[r1] * (-2 + 3/2*R[r1]) * dx0[r1] / h[r1]**2
    dwdx1[r1] = alpha[r1] * (-2 + 3/2*R[r1]) * dx1[r1] / h[r1]**2
    dwdx2[r1] = alpha[r1] * (-2 + 3/2*R[r1]) * dx2[r1] / h[r1]**2
    dwdx0[r2] = - alpha[r2] * 0.5 * (2 - R[r2])**2 * dx0[r2] / h[r2] / distance[r2]
    dwdx1[r2] = - alpha[r2] * 0.5 * (2 - R[r2])**2 * dx1[r2] / h[r2] / distance[r2]
    dwdx2[r2] = - alpha[r2] * 0.5 * (2 - R[r2])**2 * dx2[r2] / h[r2] / distance[r2]
    
    dG = np.zeros((total**2, 1))
    dG[r1] = 1/h[r1]**2 * (4/3*R[r1] - 6/5*R[r1]**3 + 0.5*R[r1]**4)
    dG[r2] = 1/h[r2]**2 * (8/3*R[r2] - 3*R[r2]**2 + 6/5*R[r2]**3 - 1/6*R[r2]**4 - 1/(15*R[r2]**2))
    dG[r3] = 1/(distance[r3]**2)

    dvdtG0 = mass * dG * dx0 / distance
    dvdtG1 = mass * dG * dx1 / distance
    dvdtG2 = mass * dG * dx2 / distance 
    
    dv =  cdist(velocity, velocity).reshape(total**2, -1)
    c = 0.5 * (np.sqrt(0.4 * energy[i]) + np.sqrt(0.4 * energy[j]))
    rho = 0.5 * (density[i] + density[j])    
    phi = np.einsum("ij,ij->j", (h * dv), distance) / (distance**2 + (0.1 * h)**2)
    PI = np.zeros((total**2, 1))    
    filtPI = (distance * dv) < 0   
    PI[filtPI] = (-c[filtPI] * phi[filtPI] + (phi[filtPI])**2 ) / rho[filtPI]

    dedt = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * np.einsum("ij,ij->i", (velocity[i] - velocity[j]), dwdx).reshape(total**2,-1)    
    dvdt0 = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * dwdx0
    dvdt1 = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * dwdx1
    dvdt2 = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * dwdx2       
    dvdtG0 = - G * np.sum(dvdtG0.reshape(total, -1), axis=1)  
    dvdtG1 = - G * np.sum(dvdtG1.reshape(total, -1), axis=1)      
    dvdtG2 = - G * np.sum(dvdtG2.reshape(total, -1), axis=1)                  
    dvdt0 = - np.sum(dvdt0.reshape(total, -1), axis=1)  
    dvdt1 = - np.sum(dvdt1.reshape(total, -1), axis=1)      
    dvdt2 = - np.sum(dvdt2.reshape(total, -1), axis=1)
    
    data[:,7] = np.sum((mass * w).reshape(total, -1), axis=1)
    data[:,8] = (0.4 * density * energy).reshape(total)    
    new = np.zeros((total, 10))
    new[:,0] = data[:,3]
    new[:,1] = data[:,4]
    new[:,2] = data[:,5]
    new[:,3] = dvdtG0 + dvdt0
    new[:,4] = dvdtG1 + dvdt1
    new[:,5] = dvdtG2 + dvdt2
    new[:,9] = 0.5 * np.sum(dedt.reshape(total, -1), axis=1)
    return new.flatten()

# %%
runtime = 2e4
evaltime = 50
time = timed()
sol = solve_ivp(FORCE, [0, runtime], data.flatten(), t_eval=np.arange(0, runtime, evaltime))
print(timed()-time)
a, b = np.shape(sol.y)
res = sol.y.reshape((total, 10, b))
# %%
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
lim = 1.5e8
scat1, = ax.plot([], [], [], marker="o", markersize=2, linestyle='None', color="teal", alpha=0.5)
scat2, = ax.plot([], [], [], marker="o", markersize=2, linestyle='None', color="crimson", alpha=0.5)
text = ax.text(-lim+0.5, lim-1, lim-1, '', fontsize=15)

def animate(i):
    scat1.set_data(res[0:len1,0,i], res[0:len1,1,i])
    scat1.set_3d_properties(res[0:len2,2,i])
    scat2.set_data(res[len1:len1+len2,0,i], res[len1:len1+len2,1,i])
    scat2.set_3d_properties(res[len1:len1+len2,2,i])
    text.set_text('{} hours'.format(np.round(sol.t[i]/(60*60), 1)))
    return scat1, scat2, text

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim([-lim,lim])
ax.set_ylim([-lim,lim])
ax.set_zlim([-lim,lim])

anim = animation.FuncAnimation(fig, animate, frames=b, interval=1)
# %%
anim.save(f'{particles}-{runtime}-{evaltime}.mp4')
# %%
fig = plt.figure(figsize=(13,10))
ax = fig.add_subplot(111, projection='3d')

lim = 2e8
ts = -1

p = ax.scatter(res[0:len1,0,ts],res[0:len1,1,ts],res[0:len1,2,ts],c=res[0:len1,7,ts],s=3,alpha=1,cmap='viridis_r')
ax.scatter(res[len1:len1+len2,0,ts],res[len1:len1+len2,1,ts],res[len1:len1+len2,2,ts],c=res[0:len1,7,ts],s=3,alpha=1,cmap='viridis_r')

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_xlim([-lim,lim])
ax.set_ylim([-lim,lim])
ax.set_zlim([-lim,lim])

cbar = fig.colorbar(p)
cbar.ax.set_ylabel(r'density (kg/m$^{-3}$)')

plt.savefig(f'close-{particles}-{runtime}-{ts}.png', bbox_inches='tight')
