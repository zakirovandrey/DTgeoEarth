#!/usr/bin/python
# -*- coding: utf-8 -*-

from ctypes import *
#mpi = CDLL('libmpi.so.0', RTLD_GLOBAL)
#mpi = CDLL('/usr/mpi/gcc/openmpi-1.4.2-qlc/lib64/libmpi.so.0', RTLD_GLOBAL)

from math import *
import sys
import os
#sys.path.append("./spacemodel")

import DTgeo

GridNx = DTgeo.cvar.GridNx
GridNy = DTgeo.cvar.GridNy
GridNz = DTgeo.cvar.GridNz
dx=DTgeo.cvar.ds
dy=DTgeo.cvar.dv
dz=DTgeo.cvar.da
dt=DTgeo.cvar.dt

class SpaceModel: pass
SM= SpaceModel()

print 'load OK'

SrcCoords_LOC  = [ GridNx/2*dx, GridNy/2*dy-6250.0, GridNz/2*dz]

SM.Vp, SM.Vs, SM.sigma = 7.92,4.42,3.37
print "Phys_params at shotpoint %g %g %g\n"%(SM.Vp,SM.Vs,SM.sigma)

SS = DTgeo.cvar.shotpoint
SS.Ampl = 0.0;
SS.F0=0.04;
SS.gauss_waist=0.5;
SS.srcXs, SS.srcXa, SS.srcXv = SrcCoords_LOC[0],SrcCoords_LOC[1],SrcCoords_LOC[2];
SS.BoxMs, SS.BoxPs = SS.srcXs-5.1*dx, SS.srcXs+5.1*dx; 
SS.BoxMa, SS.BoxPa = SS.srcXa-5.1*dy, SS.srcXa+5.1*dy; 
SS.BoxMv, SS.BoxPv = SS.srcXv-5.1*dz, SS.srcXv+5.1*dz; 
SS.sphR = 3*dz; SS.BoxMs, SS.BoxPs = SS.srcXs-SS.sphR-5*dx, SS.srcXs+SS.sphR+5*dx;
boxDiagLength=sqrt((SS.BoxPs-SS.BoxMs)**2+(SS.BoxPa-SS.BoxMa)**2+(SS.BoxMv-SS.BoxPv)**2)
SS.tStop = boxDiagLength/2/min(SM.Vp,0.0001+SM.Vs)+8/(pi*SS.F0)+10*dt # 5000*dt; # ((BoxPs-BoxMs)+(BoxPa-BoxMa)+(BoxMv-BoxPv))/c+2*M_PI/Omega;
SS.V_max = 15.0/2;
SS.start = 0;

SS.set(SM.Vp, SM.Vs, SM.sigma)

DTgeo.cvar.Tsteps=5000
DTgeo._main(sys.argv)
