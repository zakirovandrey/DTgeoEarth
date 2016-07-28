#ifndef SURFCUT_CUH
#define SURFCUT_CUH
#define EARTH_Center_X (Np*NDT/2+0*1200/dx)
#define EARTH_Center_Y (Na*NasyncNodes*NDT/2+0*100000/dy)
#define EARTH_Center_Z Nz/2

inline __host__ __device__ ftype get_radius(ftype x, ftype y, ftype z, int PStype) {
  return 63700000;//6368+100000; 
  //return 6370;//6368+100000; 
}

//CSIZE=1 or 3
//return normalized area and volume 
extern __device__ ftype get_area(const int CSIZE, const int ix, const int iy, const int iz, const int dir, int PStype);
extern __device__ ftype2 get_vols(const int ix, const int iy, const int iz, int PStype);
__device__ __forceinline__ ftype get_surf(const int CSIZE, const int ix, const int iy, const int iz, const int dir, int PStype, int Ind){
  return 1;
  if(Ind==0) return 1; else
  if(Ind==2) return 0; else {
  return get_area(abs(CSIZE),ix,iy,iz,dir,PStype);
  int crd_left[3] = {ix,iy,iz};
  int crd_rght[3] = {ix,iy,iz};
  crd_left[dir]--;
  crd_rght[dir]++;
  if(abs(CSIZE)!=1) crd_left[dir]-=2;
  if(abs(CSIZE)!=1) crd_rght[dir]+=2;
  const ftype2 vol_left = get_vols(crd_left[0],crd_left[1],crd_left[2],0);
  const ftype2 vol_rght = get_vols(crd_rght[0],crd_rght[1],crd_rght[2],0);
  const ftype2 min_vol = make_ftype2(min(vol_left.x,vol_rght.x),min(vol_left.y,vol_rght.y));
  const ftype s_area=get_area(abs(CSIZE),ix,iy,iz,dir,PStype);
  //if(s_area>=0.5 && min_vol<0.5*s_area) return min(2*min_vol,s_area);
  //if(s_area< 0.5 && min_vol<0.5*s_area) return max(s_area-0.5,0.0);
  /*if(abs(CSIZE)==1) return min(2*min_vol.x,2*max(s_area-0.5,0.0));
  else              return min(2*min_vol.y,2*max(s_area-0.5,0.0));*/
  if(abs(CSIZE)==1) return min(2*min_vol.x,s_area);
  else              return min(2*min_vol.y,s_area);
  }
}
#define SURFACE(ix,iy,iz) 1
/*__device__ __forceinline__ ftype2 get_dvols(const int ix, const int iy, const int iz, int svtype, int ind){
  if(!SURFACE(ix, iy, iz)) return make_ftype2(8./9.,0.);
  if(ind==0) return make_ftype2(8./9.,0); else
  if(ind==2) return make_ftype2(0,0);
  const ftype2 vols = get_vols(ix,iy,iz,0);
  ftype vol_small = vols.x;
  ftype vol_big   = vols.y;
  if(svtype==1) {
    if(vol_small==0) return make_ftype2(0,0);
    //if(vol_small<1)  return make_ftype2(8./9.+1./9.*vol_small, vol_small);
    if(vol_small!=0) vol_small=8./9.;
    if(vol_big  !=0) return make_ftype2(8./9., 0);
    return make_ftype2(8./9., 0);
  } else {
    vol_big=0;
//    ftype max_surf = 0;
//    max_surf = max(max_surf,get_area(1,ix-1,iy,iz,0,0));
//    max_surf = max(max_surf,get_area(1,ix+1,iy,iz,0,0));
//    max_surf = max(max_surf,get_area(1,ix,iy-1,iz,1,0));
//    max_surf = max(max_surf,get_area(1,ix,iy+1,iz,1,0));
//    max_surf = max(max_surf,get_area(1,ix,iy,iz-1,2,0));
//    max_surf = max(max_surf,get_area(1,ix,iy,iz+1,2,0));
//    vol_small = max(vol_small,0.5*max_surf);
    if(vol_small>=0.001) vol_small = 1.0/vol_small; else if(vol_small>0) vol_small=1000; else vol_small=0;
    //if(vol_small>=0.001) vol_small = 1.0/vol_small*exp(-1/vol_small*0.01)*exp(0.01); else if(vol_small>0) vol_small=0*100; else vol_small=0;
    return make_ftype2(vol_small*8./9., 0);
  }
}*/
__device__ __forceinline__ ftype2 get_dvols(const int ix, const int iy, const int iz, int svtype, int ind){
  if(!SURFACE(ix, iy, iz)) return make_ftype2(1,1);
  if(ind==0) return make_ftype2(1,1); else
  if(ind==2) return make_ftype2(0,0);
  const ftype2 vols = get_vols(ix,iy,iz,0);
  ftype vol_small = vols.x;
  ftype vol_big   = vols.y;
  if(svtype==1) {
    
if(vol_small<0.5) return make_ftype2(0,0); else return make_ftype2(1,1);

    //return make_ftype2(1, 1);
    if(vol_small==0) return make_ftype2(0,0);
    //if(vol_small<1)  return make_ftype2(8./9.+1./9.*vol_small, vol_small);
    if(vol_small!=0) vol_small=1;
    if(vol_big  !=0) return make_ftype2(1, 1);
    return make_ftype2(1, 0);
  } else {
    
return make_ftype2(1,1);

    if(vol_small==0) vol_big=0; 
    if(vol_big  >=1e-2f) vol_big   = 1.0/vol_big  ; else if(vol_big  >0) vol_big  =1e2f; else vol_big  =0;
    if(vol_small>=1e-2f) vol_small = 1.0/vol_small; else if(vol_small>0) vol_small=1e2f; else vol_small=0;
    return make_ftype2(vol_small, vol_big);

    if(vol_small==1 && vol_big<=1)    return make_ftype2(1.0/vol_small, 1.0/max(vol_big,0.2));
    if(vol_small<1 && vol_small>=0.5) return make_ftype2(1.0/vol_small*8./9., 0);
    if(vol_small<0.5 && vol_big!=0)   return make_ftype2(0, -8*1.0/max(vol_big,0.2));
    return make_ftype2(0, 0);

    //if(vol_small>=0.1) vol_small = 1.0/vol_small; else if(vol_small>0) { vol_small=0; vol_big*=-8; } else { vol_small=0;  vol_big*=-8; }
    
    //if(vol_big  >=0.01 && vol_small==1) vol_big   = 1.0/vol_big  ; else if(vol_big  >0 && vol_small==1 || vol_big<0.01 && vol_big>0) vol_big  =100;
    //else if(vol_small>0) { vol_big = (9*vol_small*vol_small-8)/vol_big; } // else if(vol_small>0) { vol_small=0; vol_big*=-8; } else vol_small=0;
    //return make_ftype2(vol_small, vol_big);
  }
}
#endif// SURFCUT_CUH
