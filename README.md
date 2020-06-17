# planetcollider
Hydrodynamical simulations of two planets colliding

### About

Hydrodynamical simulations are common in astronomy in cases where there are too many particles in the system for a traditional N-body code to be practical. In normal N-body codes every particle in the simulation is affected by every other particle. This works fine for small numbers of particles, but becomes too computationally expensive when large amounts are involved. An example of this are gas clouds. Smoothed-particle hydrodynamics solves this problem by using an algorithm to ensure that only the particles closest to the one in question can influence its behaviour. This significantly reduces the amount of calculations needed, and is generally an accurate approximation. This code is a demonstration of smoothed-particle hydrodynamics. Two data sets of particles with positions, velocities, density, and pressure are used to represent two planets. The planets are given inital conditions that ensure a collision, and then the differential equations that govern the changes in velocity, density, pressure and energy are integrated forward by steps. The resulting data is then animated in 3D.

### Getting started

Select the amount of particles to use by setting the variable `planet` equal to 300, 600, 1200 or 2400. This will select the correct data file. Start with 300 particles, as once more particles are used, the simulation time greatly increases, with the largest planets taking many hours. The `runtime` can also be changed, and a small value should be tried first, as different machines can take different amounts of time to carry out the integrations. The position and velocity of the second planet can be adjusted for different kinds of collisions. The magnitude of the spin of the planets can also be changed.

### Requirements

numpy, itertools, matplotlib, scipy, mpl_toolkits
