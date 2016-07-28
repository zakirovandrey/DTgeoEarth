#include "params.h"
#include "surfcut.cuh"
#ifdef MPI_ON
#include <mpi.h>
#endif

template<class T> __device__ inline void swap(T& a, T& b) { T tmp=a; a=b; b=tmp; }
#include "cuda_math_double.h"

inline __device__ ftype3 get_norm(const ftype& x, const ftype& y, const ftype& z, int PStype) {
  ftype3 cc = make_ftype3(x,y,z);
  const ftype3 Earth_Center = make_ftype3(EARTH_Center_X*dx, EARTH_Center_Y*dy, EARTH_Center_Z*dz);
  return normalize(cc-Earth_Center);
}
inline __device__ int get_pos_mat(ftype3 c, ftype r){
  if(dot(c,c)<r*r) return 1;
  else return 0;
}
const ftype nerr = 1e-5;
inline __device__ ftype calc_area(const ftype na, const ftype nb, const ftype gamma){
  // na>=nb>=0
  if(gamma<=0) return 0;
  if(nb<nerr) return gamma;
  if(gamma<nb) return 0.5*gamma*gamma/(na*nb);
  else return (gamma-0.5*nb)/na;
}
inline __device__ ftype calc_volume(const ftype nx, const ftype ny, const ftype nz, const ftype gam){
  if(gam<=0) return 0;
  if(ny<nerr) return gam; else
  if(nz<nerr && gam<ny+nz ) return gam*gam/(2*nx*ny); else
  if(nz<nerr && gam>=ny+nz) return (gam*gam-(gam-ny)*(gam-ny))/(2*nx*ny); else
    
  if(0 <=gam && gam<nz) return gam*gam*gam/(6*nx*ny*nz); else
  if(nz<=gam && gam<ny) return (gam*gam*gam-(gam-nz)*(gam-nz)*(gam-nz))/(6*nx*ny*nz); else
  if(ny<=gam && gam<nx && gam<ny+nz ) return (gam*gam*gam-(gam-nz)*(gam-nz)*(gam-nz)-(gam-ny)*(gam-ny)*(gam-ny))/(6*nx*ny*nz); else
  if(ny<=gam && gam<nx && gam>=ny+nz) return (gam*gam*gam-(gam-nz)*(gam-nz)*(gam-nz)-(gam-ny)*(gam-ny)*(gam-ny)+(gam-ny-nz)*(gam-ny-nz)*(gam-ny-nz))/(6*nx*ny*nz); else
  if(nx<=gam) return (gam*gam*gam-(gam-nz)*(gam-nz)*(gam-nz)-(gam-ny)*(gam-ny)*(gam-ny)-(gam-nx)*(gam-nx)*(gam-nx))/(6*nx*ny*nz); else
  return 0;
}
//CSIZE=1 or 3
//return normalized area and volume 
__device__ ftype get_area(const int CSIZE, const int ix, const int iy, const int iz, const int dir, int PStype){
  ftype r = get_radius(ix*dx*0.5,iy*dy*0.5,iz*dz*0.5, PStype);
  ftype3 norm = get_norm(ix*dx*0.5,iy*dy*0.5,iz*dz*0.5, PStype);
      ftype norms[3] = {norm.x,norm.y,norm.z};
      ftype normdir = norms[dir];

  ftype3 cellcnt = make_ftype3(ix*dx*0.5, iy*dy*0.5, iz*dz*0.5);
  const ftype3 Earth_Center = make_ftype3(EARTH_Center_X*dx, EARTH_Center_Y*dy, EARTH_Center_Z*dz);
  const ftype dist2cnt = length(cellcnt-Earth_Center);
  ftype Ec[3] = {EARTH_Center_X*dx, EARTH_Center_Y*dy, EARTH_Center_Z*dz};

  ftype3 lcv = cellcnt-Earth_Center; ftype lc = sqrtf(lcv.x*lcv.x*(dir!=0)+lcv.y*lcv.y*(dir!=1)+lcv.z*lcv.z*(dir!=2));
  ftype cc[3] = {ix*dx*0.5, iy*dy*0.5, iz*dz*0.5};
  ftype l0 = sqrtf(r*r-(cc[dir]-Ec[dir])*(cc[dir]-Ec[dir])); if(isnan(l0)) l0=0;
  ftype delta;
  if(l0<100*dx) delta = lc-l0;
  else          delta = (dist2cnt+r)/(lc+l0)*(dist2cnt-r);

  int pos_in = get_pos_mat(cellcnt, r);

  ftype nrm[3] = {norm.x,norm.y,norm.z};
  ftype na = nrm[(dir+1)%3];
  ftype nb = nrm[(dir+2)%3];
  ftype nl = (na*na+nb*nb);
  if(nl==0 &&  pos_in) return 1; else
  if(nl==0 && !pos_in) return 0;
  na = fabs(na); nb = fabs(nb);
  if(na<nb) swap(na,nb);
  const float rsqnl=rsqrtf(nl);
  na*=rsqnl; nb*=rsqnl;
  delta/=dx;
  ftype gamma = 0.5*(na+nb)-fabs(delta)/CSIZE;
  ftype S = calc_area(na,nb,gamma);
  if(delta<0) S = 1-S;
  //if(abs(iz-EARTH_Center_Z*2)<3 && abs(ix-EARTH_Center_X*2)<3) printf("normdir=%.9f dir=%d cellcnt(%g %g %g) Ec(%g %g %g) na=%.9f nb=%.9f delta=%.9f S=%.9f ix-xc iy iz-zc=%d %d %d\n",normdir, dir,cellcnt.x,cellcnt.y,cellcnt.z,Ec[0],Ec[1],Ec[2],na,nb,delta,S, ix-EARTH_Center_X*2,iy,iz-EARTH_Center_Z*2);
  //if(abs(iz-EARTH_Center_Z*2)<3 && abs(ix-EARTH_Center_X*2)<3) printf("iy ix-CNT iz-CNT %d %d %d dir=%d, CSIZE=%d, S=%g\n",iy,ix-EARTH_Center_X*2,iz-EARTH_Center_Z*2,dir,CSIZE,S);
  return S;
}
__device__ ftype2 get_vols(const int ix, const int iy, const int iz, int PStype){
  ftype r = get_radius(ix*dx*0.5,iy*dy*0.5,iz*dz*0.5, PStype);
  ftype3 norm = get_norm(ix*dx*0.5,iy*dy*0.5,iz*dz*0.5, PStype);
  ftype3 cellcnt = make_ftype3(ix*dx*0.5, iy*dy*0.5, iz*dz*0.5);
  const ftype3 Earth_Center = make_ftype3(EARTH_Center_X*dx, EARTH_Center_Y*dy, EARTH_Center_Z*dz);
  ftype delta = length(cellcnt-Earth_Center)-r;
  delta/=dx;
  ftype nx = fabs(norm.x);
  ftype ny = fabs(norm.y);
  ftype nz = fabs(norm.z);
  if(nz>ny) swap(nz, ny);
  if(nz>nx) swap(nz, nx);
  if(ny>nx) swap(ny, nx);
  ftype gamma1 = 0.5*(nx+ny+nz)-fabs(delta);
  ftype gamma3 = 0.5*(nx+ny+nz)-fabs(delta)/3;
  ftype V1 = calc_volume(nx,ny,nz,gamma1);
  ftype V3 = calc_volume(nx,ny,nz,gamma3);
  if(delta<0) { V1 = 1-V1; V3 = 1-V3; }
  //if(abs(iz-EARTH_Center_Z*2)<3 && abs(ix-EARTH_Center_X*2)<3) printf("iy ix-CNT iz-CNT %d %d %d V1,V3=%g %g\n",iy,ix-EARTH_Center_X*2,iz-EARTH_Center_Z*2,V1,V3);
  return make_ftype2(V1,V3);
}

