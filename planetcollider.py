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
particles = 600
# import file which contains x,y,z,vx,vy,vz,mass,density,pressure for each particle
planet = np.loadtxt(f'./data/Planet{particles}.dat')

# roughly estimate radius of planet by taking maximum x position value
r = np.amax(planet[:,0])

# initiate planet with spin by adding perpendicular velocity to each particle
spin = 0.0003 # magnitude of spin
planet[:,3] = -planet[:,1] * spin
planet[:,4] = planet[:,0] * spin

# create a second planet by copying the first
planet2 = np.copy(planet)

planet2[:,1] += 5*r       # move second planet away in the y direction
planet2[:,0] += -0.3*r       # move second planet away in the x direction
planet2[:,4] += -2e4       # give second planet a velocity towards first to ensure collision

data = np.vstack((planet, planet2)) # add planets into one data set for integrator
# data = planet # use this to just simulate one planet

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
    
    
    h = np.full((total**2, 1),1.3e7) # create array of constant smoothing length h - particles are only influenced by others closer than h
    dx = position[i] - position[j] # difference in position between each particle and every other particle - not same as distance! vector, not scalar
    
    distance = cdist(position, position).reshape(total**2, -1) # distance between each particle and every other particle
    R = distance / h                    # distance divided by smoothing length for each pair of particles - needed in smoothing function
    alpha = 3 / (2 * np.pi * h**3)      # alpha takes this value in smoothing function in 3D - needed in smoothing function
    h[h==0]=1e-9                        # replace zero values in h with small value to prevent crash
    distance[distance==0]=1e-9          # replace zero values in distance with small value to prevent crash
    R[R==0]=1e-9                        # replace zero values in R with small value to prevent crash
    
    # masks for different values of R in cubic smoothing function
    r1 = R < 1
    r2 = np.logical_and(R >= 1, R < 2)
    r3 = R >= 2
    
    w = np.zeros((total**2, 1))                             # initialize emtpy array for results from cubic smoothing function
    w[r1] = alpha[r1] * (2/3 - R[r1]**2 + 0.5 * R[r1]**3)   # cubic smoothing function for 0 <= R < 1
    w[r2] = alpha[r2] * 1/6 * (2 - R[r2])**3                # cubic smoothing function for 1 <= R < 2
    
    dwdx = np.zeros((total**2, 3))                          # empty array for derivative of smoothing function
    dx0 = dx[:,0].reshape(total**2, -1)                     # x values of distance between particles - separated to enable masking in 1D
    dx1 = dx[:,1].reshape(total**2, -1)                     # y values of distance between particles - separated to enable masking in 1D
    dx2 = dx[:,2].reshape(total**2, -1)                     # z values of distance between particles - separated to enable masking in 1D
    dwdx0 = dwdx[:,0].reshape(total**2, -1)                 # x values of empty dwdx
    dwdx1 = dwdx[:,1].reshape(total**2, -1)                 # y values of empty dwdx
    dwdx2 = dwdx[:,2].reshape(total**2, -1)                 # z values of empty dwdx
    dwdx0[r1] = alpha[r1] * (-2 + 3/2*R[r1]) * dx0[r1] / h[r1]**2       # x values of derivative of smoothing function for R < 1
    dwdx1[r1] = alpha[r1] * (-2 + 3/2*R[r1]) * dx1[r1] / h[r1]**2       # y values of derivative of smoothing function for R < 1
    dwdx2[r1] = alpha[r1] * (-2 + 3/2*R[r1]) * dx2[r1] / h[r1]**2       # z values of derivative of smoothing function for R < 1
    dwdx0[r2] = -alpha[r2] * 0.5 * (2 - R[r2])**2 * dx0[r2] / h[r2] / distance[r2] # x values of derivative of smoothing function for 1 >= R < 2
    dwdx1[r2] = -alpha[r2] * 0.5 * (2 - R[r2])**2 * dx1[r2] / h[r2] / distance[r2] # y values of derivative of smoothing function for 1 >= R < 2
    dwdx2[r2] = -alpha[r2] * 0.5 * (2 - R[r2])**2 * dx2[r2] / h[r2] / distance[r2] # z values of derivative of smoothing function for 1 >= R < 2
    
    dG = np.zeros((total**2, 1))                                        # empty array for derivative of gravitational potential
    dG[r1] = 1/h[r1]**2 * (4/3*R[r1] - 6/5*R[r1]**3 + 0.5*R[r1]**4)     # derivative of gravitational potential for R < 1
    dG[r2] = 1/h[r2]**2 * (8/3*R[r2] - 3*R[r2]**2 + 6/5*R[r2]**3 - 1/6*R[r2]**4 - 1/(15*R[r2]**2)) # derivative of gravitational potential for 1 >= R < 2
    dG[r3] = 1/(distance[r3]**2)                                        # derivative of gravitational potential for R >= 2

    dvdtG0 = mass * dG * dx0 / distance                                 # x values of the change in velocity due to gravity
    dvdtG1 = mass * dG * dx1 / distance                                 # y values of the change in velocity due to gravity
    dvdtG2 = mass * dG * dx2 / distance                                 # z values of the change in velocity due to gravity
    
    # requirements for artificial viscosity
    dv =  cdist(velocity, velocity).reshape(total**2, -1)               # relative velocities between each particle
    c = 0.5 * (np.sqrt(0.4 * energy[i]) + np.sqrt(0.4 * energy[j]))     # average sound speed in system for each pair of particles
    rho = 0.5 * (density[i] + density[j])                               # average density of each pair of particles
    phi = np.einsum("ij,ij->j", (h * dv), distance) / (distance**2 + (0.1 * h)**2) # einsum needed to calculate phi
    PI = np.zeros((total**2, 1))                                        # initialize array for artificial viscosity
    filtPI = (distance * dv) < 0                                        # only calculate viscosity for particle pairs where distance times dv is negative
    PI[filtPI] = (-c[filtPI] * phi[filtPI] + (phi[filtPI])**2 ) / rho[filtPI] # equation for artificial viscosity

    # change in energy for each particles
    dedt = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * np.einsum("ij,ij->i", (velocity[i] - velocity[j]), dwdx).reshape(total**2,-1)    
    dvdt0 = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * dwdx0 # change in x velocity due to pressure and density
    dvdt1 = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * dwdx1 # change in y velocity due to pressure and density
    dvdt2 = mass * (pressure[i] / density[i]**2 + pressure[j] / density[j]**2 + PI) * dwdx2 # change in z velocity due to pressure and density       
    dvdtG0 = -G * np.sum(dvdtG0.reshape(total, -1), axis=1)                                 # sum of change in x velocity due to gravity
    dvdtG1 = -G * np.sum(dvdtG1.reshape(total, -1), axis=1)                                 # sum of change in y velocity due to gravity
    dvdtG2 = -G * np.sum(dvdtG2.reshape(total, -1), axis=1)                                 # sum of change in z velocity due to gravity
    dvdt0 = -np.sum(dvdt0.reshape(total, -1), axis=1)                                       # sum of change in x velocity due to pressure and density
    dvdt1 = -np.sum(dvdt1.reshape(total, -1), axis=1)                                       # sum of change in y velocity due to pressure and density
    dvdt2 = -np.sum(dvdt2.reshape(total, -1), axis=1)                                       # sum of change in z velocity due to pressure and density
    
    data[:,7] = np.sum((mass * w).reshape(total, -1), axis=1)       # new density for each particle - not put through integrator
    data[:,8] = (0.4 * density * energy).reshape(total)             # new pressure for each particle - not put through integrator
    new = np.zeros((total, 10))
    new[:,0] = data[:,3]                                            # new x position from old x velocity
    new[:,1] = data[:,4]                                            # new y position from old y velocity
    new[:,2] = data[:,5]                                            # new z position from old z velocity
    new[:,3] = dvdtG0 + dvdt0                                       # new x velocity from addition of vx from gravity and density and pressure
    new[:,4] = dvdtG1 + dvdt1                                       # new y velocity from addition of vy from gravity and density and pressure
    new[:,5] = dvdtG2 + dvdt2                                       # new z velocity from addition of vz from gravity and density and pressure
    new[:,9] = 0.5 * np.sum(dedt.reshape(total, -1), axis=1)        # new energy for each particle
    
    return new.flatten()

# %%
runtime = 3.5e4
evaltime = 50
time = timed()
# This function numerically integrates a system of ordinary differential equations given an initial value - only takes flattened arrays
sol = solve_ivp(FORCE, [0, runtime], data.flatten(), t_eval=np.arange(0, runtime, evaltime)) # t_eval is optional and select the times to store solution
print(timed()-time)
a, b = np.shape(sol.y)
res = sol.y.reshape((total, 10, b)) # reshape for plotting
# %%
fig = plt.figure(figsize=(10,10))
ax = fig.add_subplot(111, projection='3d')
lim = 12e4
planetscat1, = ax.plot([], [], [], marker="o", markersize=8, linestyle='None', color="darkblue", alpha=0.3)
planetscat2, = ax.plot([], [], [], marker="o", markersize=8, linestyle='None', color="darkred", alpha=0.3)
text = ax.text(-lim+0.5, lim-1, lim-1, '', fontsize=15)

def animate(i):
    planetscat1.set_data(res[0:len1,0,i]/1e3, res[0:len1,1,i]/1e3)
    planetscat1.set_3d_properties(res[0:len2,2,i]/1e3)
    planetscat2.set_data(res[len1:len1+len2,0,i]/1e3, res[len1:len1+len2,1,i]/1e3)
    planetscat2.set_3d_properties(res[len1:len1+len2,2,i]/1e3)
    text.set_text('{} hours'.format(np.round(sol.t[i]/(60*60), 1)))
    return planetscat1, planetscat2, text

ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_xlim([-lim,lim])
ax.set_ylim([-lim,lim])
ax.set_zlim([-lim,lim])

im_ani = animation.FuncAnimation(fig, animate, frames=b, interval=1)
# %%
plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

f = f'vid/double_planet1_{particles}.mp4' 
writervideo = animation.FFMpegWriter(fps=40) # ffmpeg must be installed
im_ani.save(f, writer=writervideo)
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

# plt.savefig(f'close-{particles}-{runtime}-{ts}.png', bbox_inches='tight')
