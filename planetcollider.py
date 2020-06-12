# %%
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from matplotlib import animation
from scipy.integrate import solve_ivp
from timeit import default_timer as timed
from scipy.spatial.distance import cdist
from mpl_toolkits.mplot3d import axes3d


# constants
G = 6.674e-11 # gravitational constant in SI units

# select which data file to import: 300, 600, 1200 or 2400
particles = 300
# import file which contains x,y,z,vx,vy,vz,mass,density,pressure for each particle
planet = np.loadtxt(f'./data/Planet{particles}.dat')

# roughly estimate radius of planet by taking maximum x position value
r = np.amax(planet[:,0])

# initiate planet with spin by adding perpendicular velocity to each particle
spin = 0.0004 # magnitude of spin
planet[:,3] = - planet[:,1] * spin
planet[:,4] = planet[:,0] * spin

# create a second planet by copying the first
planet2 = np.copy(planet)
# move second planet away in the y direction
planet2[:,1] += 1.5*r
# move second planet away in the x direction
planet2[:,0] += - 1*r
# give second planet a velocity towards first to ensure collision
planet2[:,4] += - 2e4
# add planets into one data set for integrator
data = np.vstack((planet, planet2))


total = len(data) # amount of particles in both planets
energy = data[:,8] / data[:,7] / 0.4  # calculate energy of particle from density and pressure
data = np.append(data, energy.reshape(total, -1), axis=1) # add energy column to data
len1, len2 = len(planet), len(planet2) # amount of particles in each planet

# use itertools.product to create combinations to avoid loops
combinations = np.array(list(product(np.arange(total), np.arange(total))))
i, j = combinations[:,0], combinations[:,1]

# function that calculates the change in properties for each particle
def FORCE(t, data):
    # integrator needs flattened array so FORCE function takes flattened array which needs to be reshaped
    data = data.reshape(total, -1)    
    # pull out different properties of particles
    position = data[:,0:3]
    velocity = data[:,3:6]
    mass = np.mean(data[:,6])
    density = data[:,7].reshape(total, -1)
    pressure = data[:,8].reshape(total, -1)
    energy = data[:,9].reshape(total, -1)
    
    
    h = np.full((total**2, 1),1.3e7) # create array of h values same length as data
    dx = position[i] - position[j] # difference in position between each particle and every other particle - length is len(data)**2
    
    distance = cdist(position, position).reshape(total**2, -1) # distance between each particle and every other particle
    R = distance / h                    # R in equation
    alpha = 3 / (2 * np.pi * h**3)      # alpha in equation
    h[h==0]=1e-9                        # replace zero values in h with small value to prevent crash
    distance[distance==0]=1e-9          # replace zero values in distance with small value to prevent crash
    R[R==0]=1e-9                        # replace zero values in R with small value to prevent crash
    
    r1 = R < 1                          # r1 in equation
    r2 = np.logical_and(R >= 1, R < 2)  # r1 in equation
    r3 = R >= 2                         # r3 in equation
    
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
