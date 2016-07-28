#include "params.h"
#include "texmodel.cuh"
//using namespace aiv;
#ifdef MPI_ON
#include <mpi.h>
#endif
#include <fstream>
#include "ubs.hpp"

__constant__ float texStretchH;
__constant__ float4 texStretch[MAX_TEXS];
__constant__ float4 texShift[MAX_TEXS];
__constant__ float4 texStretchShow;
__constant__ float4 texShiftShow;
#ifdef USE_TEX_REFS
texture<coffS_t, cudaTextureType3D, cudaReadModeElementType> layerRefS;
texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefV;
texture<float2 , cudaTextureType3D, cudaReadModeElementType> layerRefQ;
texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefT;
texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefTa;
texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefTi;
#endif
texture<coffS_t, cudaTextureType3D, cudaReadModeElementType> radTexS;
texture<float  , cudaTextureType3D, cudaReadModeElementType> radTexV;
texture<float  , cudaTextureType3D, cudaReadModeElementType> radTexT;
texture<float2 , cudaTextureType3D, cudaReadModeElementType> radTexQ;
void get_mat(float r, ftype& Vp, ftype& Vs, ftype& rho, ftype2& Q){
//  ftype x0 = Np*NDT*dx/2;
//  ftype y0 = Na*NDT*dy/2;
//  ftype z0 = Nz*dz/2;
//  ftype r = sqrt((x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0));
  ftype smth = 3*dx;
  if(r<1200) { Vp=defCoff::Vp; Vs=defCoff::Vs; rho=defCoff::rho;}
  else if(r<1200+smth) { 
    ftype delta=r-1200;
    Vp=defCoff::Vp+(delta/smth)*(delta/smth)*((defCoff::Vp+0)*0.5-defCoff::Vp);
    Vs=defCoff::Vs+(delta/smth)*(delta/smth)*((defCoff::Vs+0)*0.5-defCoff::Vs);
    //rho=defCoff::rho+(delta/smth)*(delta/smth)*((defCoff::rho+0.1)*0.5-defCoff::rho);
    if(Vp<0) Vp=0;
    if(Vs<0) Vs=0;
  }
  else if(r<1200+2*smth) { 
    ftype delta=1200+2*smth-r;
    Vp=defCoff::Vp;
    Vs=defCoff::Vs;
    Vp=0+(delta/smth)*(delta/smth)*((defCoff::Vp+0)*0.5-0);
    Vs=0+(delta/smth)*(delta/smth)*((defCoff::Vs+0)*0.5-0);
    //rho=0.1+(delta/smth)*(delta/smth)*((defCoff::rho+0.1)*0.5-0.1);
    if(Vp<0) Vp=0;
    if(Vs<0) Vs=0;
  }
  else {Vp=defCoff::Vp*0; Vs=defCoff::Vs*0; rho=defCoff::rho;}
  //Vp=z; Vs=1; rho=1;
}
void get_mat_tex(int ir, ftype& Vp, ftype& Vs, ftype& rho, ftype2& Q){
  Vp = parsHost.texs.Vp[ir];
  Vs = parsHost.texs.Vs[ir];
  rho= parsHost.texs.Rho[ir];
  Q  = parsHost.texs.Qg[ir];
}
namespace lev{
        inline double sign(double x) { return (x<0.)?-1.:1.; }
        inline double Lambda3m1(double x) { //-2<x<2; -0.5<f<0.5
                const double ax = fabs(x), sx = sign(x);
                if( ax>=2. ) return 0.5*sx;
                const double xx = x*x, xxx = x*xx, /*xxxx = xx*xx,*/ x1 = x, x2 = ax*x, x3 = xxx, x4 = ax*xxx;
                if( ax>=1. ) return -(1./6.)*sx+(4./3.)*x1-1*x2+(1./3.)*x3-(1./24.)*x4;
                return (2./3.)*x1-(1./3.)*x3+(1./8.)*x4;
        }
        double Lambda5(double x) {
                double sc = 3. ; //M_PI;//*sqrt(2./3.);
                x*= sc; //(sc*x-3.);
                const double ax=fabs(x);
                if(ax>=3.) return 0.0;
                const double xx=x*x, xxxx=xx*xx, x1=ax, x2=xx, x3=ax*xx, x4=xxxx, x5=ax*xxxx;
                if(ax>=2.) return (81./40.-(27./8.)*x1+(9./4.)*x2-(3./4.)*x3+(1./8.)*x4-(1./120.)*x5)/(sc*sc);
                if(ax>=1.) return (17./40.+(5./8.)*x1-(7./4.)*x2+(5./4.)*x3-(3./8.)*x4+(1./24.)*x5)/(sc*sc);
                return (11./20.-(1./2.)*x2+(1./4.)*x4-(1./12.)*x5)/(sc*sc);
        }
        inline double Psi(double x, double x1, double x2){ // -x1<x<x2, x1,2>0 ==> [0:1]
                if(x<0) return 2*x1/(x1+x2)*(.5+Lambda3m1(2*x/x1));
                else return (x1+2*x2*(Lambda3m1(2*x/x2)))/(x1+x2);
        }

}
inline int2 get_sphere(double x,double y,double z){
  double x0 = Np*NDT*dx/2;
  double y0 = Na*NDT*dy/2+6100*0;
  double z0 = Nz*dz/2;
  double r2 = (x-x0)*(x-x0) + (y-y0)*(y-y0) + (z-z0)*(z-z0);
  double cutr = 6368;
  double cutK = 3500;
  double cutI = 1216;
  if(r2>=cutr*cutr)  return make_int2(0,0); else
  if(r2<=cutK*cutK && r2>=cutI*cutI) return make_int2(1,0);
  else return make_int2(1,1);
};
float4 get_mat(double x, double y, double z){
  return make_float4(0,0,0,0);
/*
  float3 cellcnt = make_ftype3(x,y,z);
  const float3 Earth_Center = make_ftype3(EARTH_Center_X*dx, EARTH_Center_Y*dy, EARTH_Center_Z*dz);
  double r = length(cellcnt-Earth_Center);
  //r = fabs(cellcnt.y-Earth_Center.y);
  ftype Vp=defCoff::Vp;
  ftype Vs=defCoff::Vs;
  ftype rho=defCoff::rho;
 
  double cutr = 6368;
  double cutr_2 = 5358;
  //if(r2<cutr*cutr) return make_float4(1.0/rho, rho*(Vp*Vp)  , rho*(Vp*Vp-2*Vs*Vs), rho*(Vs*Vs));
  //else             return make_float4(1.0/rho*0, rho*(Vp*Vp), rho*(Vp*Vp-2*Vs*Vs), rho*(Vs*Vs));

  if(r>=6600-1) r=6600-1;
  rho = interpolate_ubs<3>(Rhot , r/40-floor(r/40), int(r/40));
  Vp  = interpolate_ubs<3>(VelPt, r/40-floor(r/40), int(r/40));
  Vs  = interpolate_ubs<3>(VelSt, r/40-floor(r/40), int(r/40));
  //rho=2.6;
  //Vp=5.8;
  //Vs=3.2;
  return make_float4(1.0/rho, rho*(Vp*Vp)  , rho*(Vp*Vp-2*Vs*Vs), rho*(Vs*Vs));

  r = VelP[0];
  ftype arg = r-6100*0;
  arg = arg/(3*dx)*2;

  ftype k1=1.0/rho;
  ftype k2=1.0/rho*0;
  ftype ksmth0 = (lev::Lambda3m1(arg)+0.5)*(k2-k1)+k1;

        k1=rho*(Vp*Vp);
        k2=rho*(Vp*Vp*0);
  ftype ksmth1 = (lev::Lambda3m1(arg)+0.5)*(k2-k1)+k1;

        k1=rho*(Vp*Vp-2*Vs*Vs);
        k2=rho*(Vp*Vp*0-2*Vs*Vs*0);
  ftype ksmth2 = (lev::Lambda3m1(arg)+0.5)*(k2-k1)+k1;

        k1=rho*(Vs*Vs);
        k2=rho*(Vs*Vs*0);
  ftype ksmth3 = (lev::Lambda3m1(arg)+0.5)*(k2-k1)+k1;
  //if(r<1300+20000) return make_float4(1.0/rho, ksmth1, ksmth2, ksmth3);
  //else               return make_float4(1.0/rho*0, ksmth1, ksmth2, ksmth3);
  return make_float4(ksmth0, ksmth1, ksmth2, ksmth3);*/
}
void ModelTexs::fill_array(float4 StretchHost, float4 ShiftHost){
  std::ifstream infile("PREM.txt");
  const int Nr=texN[0].z;
  const int scale=3;
  const int Nbig=ceil(Nr/float(scale));
  float* bVp = new float[Nbig]; float* bVs = new float[Nbig]; float* bRho = new float[Nbig]; float* bQgx= new float[Nbig]; float* bQgy= new float[Nbig];
  memset(bVp , 0, Nbig*sizeof(float));
  memset(bVs , 0, Nbig*sizeof(float));
  memset(bRho, 0, Nbig*sizeof(float));
  memset(bQgx, 0, Nbig*sizeof(float));
  memset(bQgy, 0, Nbig*sizeof(float));
  float rho,dist,vp,vs,q;
  int tex_crd_prev=-1000; float vp_prev=0,vs_prev=0,rho_prev=0;
  while(infile >> dist >> rho >> vp >> vs){
    int dmesh_crd = int(round(dist/(dz*0.5)));
    int tex_crd = (dmesh_crd*StretchHost.z+ShiftHost.z)*texN[0].z;
    if(tex_crd>=Nr) tex_crd=Nr-1;
    if(tex_crd_prev==-1000) tex_crd_prev=tex_crd-1;
    if(tex_crd>tex_crd_prev) {
      for(int itexc=tex_crd_prev+1; itexc<=tex_crd; itexc++) {
        double alpha=double(itexc-tex_crd_prev)/double(tex_crd-tex_crd_prev);
        bVp [itexc/scale] = vp_prev *(1-alpha)+vp *alpha;
        bVs [itexc/scale] = vs_prev *(1-alpha)+vs *alpha;
        bRho[itexc/scale] = rho_prev*(1-alpha)+rho*alpha;
        printf("indB=%d bVp=%g\n",itexc/scale,bVp[itexc/scale]);
      }
    }
    else if(tex_crd<tex_crd_prev) {
      for(int itexc=tex_crd_prev-1; itexc>=tex_crd; itexc--) {
        double alpha=double(-itexc+tex_crd_prev)/double(tex_crd_prev-tex_crd);
        bVp [itexc/scale] = vp_prev *(1-alpha)+vp *alpha;
        bVs [itexc/scale] = vs_prev *(1-alpha)+vs *alpha;
        bRho[itexc/scale] = rho_prev*(1-alpha)+rho*alpha;
        printf("indB=%d bVp=%g\n",itexc/scale,bVp[itexc/scale]);
      }
    }
    tex_crd_prev=tex_crd;
    vp_prev=vp; vs_prev=vs; rho_prev=rho;
  }
  float rho_bnd=1.0;
  for(int ir=0; ir<Nbig; ir++) {
    float2 q = make_float2(0,0);
    float2 qs = make_float2(0,0);
    if(bVp[ir]==0) { qs.x=2/dt; bRho[ir]=rho_bnd; } else rho_bnd=bRho[ir];
    if(bVs[ir]==0) qs.y=2/dt;
    bQgx[ir] = (2-qs.x*dt)/(2+qs.x*dt);
    bQgy[ir] = (2-qs.y*dt)/(2+qs.y*dt);
    printf("ir=%d Vp=%g Qq=%g %g\n",ir,bVp[ir],bQgx[ir],bQgy[ir]);
  }
  //----------------------integpolete by sergey's ubs fubnction----------------------//

  Vp = new float[Nr]; Vs = new float[Nr]; Rho = new float[Nr]; Qg= new float2[Nr];
  memset(Vp , 0, Nbig*sizeof(float));
  memset(Vs , 0, Nbig*sizeof(float));
  memset(Rho, 0, Nbig*sizeof(float));
  memset(Qg, 0, Nbig*sizeof(float2));
  for(int ir=0; ir<Nr; ir++) {
    Vp[ir] = interpolate_ubs<3>(bVp , (ir%scale)/double(scale), ir/scale);
    Vs[ir] = interpolate_ubs<3>(bVs , (ir%scale)/double(scale), ir/scale);
    Rho[ir]= interpolate_ubs<3>(bRho, (ir%scale)/double(scale), ir/scale);
    Qg[ir].x=interpolate_ubs<3>(bQgx, (ir%scale)/double(scale), ir/scale);
    Qg[ir].y=interpolate_ubs<3>(bQgy, (ir%scale)/double(scale), ir/scale);
  }
  delete[] bVp; delete[] bVs; delete[] bRho; delete[] bQgx; delete[] bQgy;

/*  float r,d,vp,vs,q;
  int rprev = -10;
  for(int i=0; i<6600; i++) { VelP[i]=0; VelS[i]=0; Rho[i]=0; Qg[i]=0; }
  for(int i=0; i<330; i++) { VelPt[i]=0; VelSt[i]=0; Rhot[i]=0; Qt[i]=0; }
  int begin=1; ftype dayVp,dayVs,dayRho,dayQ; int Rmax;
  while(infile >> r >> d >> vp >> vs >> q){
    if(begin>0) {Rmax=int(r); dayVp=vp; dayVs=vs; dayRho=d; dayQ=q; begin=0;}
    int rI = int(r);
    if(rprev<0) rprev=rI+1;
    int rcur=rprev-1;
    ftype kVp = (vp-VelP[rprev])/(rcur-rI+1);
    ftype kVs = (vs-VelS[rprev])/(rcur-rI+1);
    ftype kR  = (d -Rho [rprev])/(rcur-rI+1);
    ftype kQ  = (q -Qg  [rprev])/(rcur-rI+1);
    while(rI<rprev-1) {
      rcur=rprev-1;
      VelP[rcur] = VelP[rprev]+kVp;
      VelS[rcur] = VelS[rprev]+kVs;
      Rho [rcur] = Rho[rprev] +kR;
      Qg  [rcur] = Qg [rprev] +kQ;
      rprev--;
    }
    rcur=rprev-1;
    if(rI==rprev) continue;
    VelP[rcur] = vp; 
    VelS[rcur] = vs; 
    Rho [rcur] = d;
    Qg  [rcur] = q;
    rprev=rcur;
  }
  for(int i=Rmax+1; i<6600; i++) { VelP[i]=dayVp; VelS[i]=dayVs; Rho[i]=dayRho; Qg[i]=dayQ; }
  for(int i=0; i<330; i++) { VelPt[i]=VelP[i*40]; VelSt[i]=VelS[i*40]; Rhot[i]=Rho[i*40]; Qt[i]=Qg[i*40]; }
  for(double dist=0; dist<6600; dist+=0.1) {
    printf("%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\t%g\n",dist,Rho[int(dist)],VelP[int(dist)],VelS[int(dist)],Qg[int(dist)],
      interpolate_ubs<3>(Rhot , dist/40-floor(dist/40), int(dist/40)),
      interpolate_ubs<3>(VelPt, dist/40-floor(dist/40), int(dist/40)),
      interpolate_ubs<3>(VelSt, dist/40-floor(dist/40), int(dist/40)),
      interpolate_ubs<3>(Qt   , dist/40-floor(dist/40), int(dist/40))
    );
  }*/
}
//texture <float2  , cudaTextureType3D, cudaReadModeElementType> volfractex;
//texture <float2  , cudaTextureType3D, cudaReadModeElementType> surfractex;

//extern __device__ ftype get_area(const int CSIZE, const int ix, const int iy, const int iz, const int dir, int PStype);
//extern __device__ ftype2 get_vols(const int ix, const int iy, const int iz, int PStype);

void fill_vol_surf_frac_array(float2* VolFracHost, float2* SurFracHost, const int texFnx, const int texFny, const int texRd){
  for(int nx=0; nx<texFnx; nx++) for(int ny=0; ny<texFny; ny++) for(int r=0; r<texRd; r++) {
  }
}
void ModelTexs::init(){
  int node=0, Nprocs=1;
  #ifdef MPI_ON
  MPI_Comm_rank (MPI_COMM_WORLD, &node);
  MPI_Comm_size (MPI_COMM_WORLD, &Nprocs);
  #endif
  //---------------------------------------------------//--------------------------------------
  ShowTexBinded=0;
  Ntexs=1; // get from aivModel
  if(Ntexs>MAX_TEXS) { printf("Error: Maximum number of texs is reached (%d>%d)\n", Ntexs, MAX_TEXS); exit(-1); }
  HostLayerS = new coffS_t*[Ntexs]; HostLayerV = new float*[Ntexs]; HostLayerQ = new float2*[Ntexs]; HostLayerT = new float*[Ntexs]; HostLayerTi = new float*[Ntexs]; HostLayerTa = new float*[Ntexs]; 
  for(int idev=0;idev<NDev;idev++) { DevLayerS[idev] = new cudaArray*[Ntexs]; DevLayerV[idev] = new cudaArray*[Ntexs]; DevLayerQ[idev] = new cudaArray*[Ntexs]; DevLayerT[idev] = new cudaArray*[Ntexs]; DevLayerTi[idev] = new cudaArray*[Ntexs]; DevLayerTa[idev] = new cudaArray*[Ntexs]; }
  for(int idev=0;idev<NDev;idev++) { layerS_host[idev] = new cudaTextureObject_t[Ntexs]; layerV_host[idev] = new cudaTextureObject_t[Ntexs]; layerQ_host[idev] = new cudaTextureObject_t[Ntexs]; layerT_host[idev] = new cudaTextureObject_t[Ntexs]; layerTi_host[idev] = new cudaTextureObject_t[Ntexs];  layerTa_host[idev] = new cudaTextureObject_t[Ntexs]; }
  for(int idev=0;idev<NDev;idev++) { CHECK_ERROR( cudaSetDevice(idev) );
    CHECK_ERROR( cudaMalloc((void**)&layerS [idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerV [idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerQ [idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerT [idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerTi[idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
    CHECK_ERROR( cudaMalloc((void**)&layerTa[idev], Ntexs*sizeof(cudaTextureObject_t)) ); 
  }
  CHECK_ERROR( cudaSetDevice(0) );
  int Nh=1; unsigned long long texsize_onhost=0, texsize_ondevs=0;
  texN    = new int3 [Ntexs];     //get from aivModel
  tex0    = new int  [Ntexs];     //get from aivModel
  texStep = new float[Ntexs];     //get from aivModel
  float4 texStretchHost[MAX_TEXS];
  float4 texShiftHost[MAX_TEXS];
  const int Nrmax=max(max(Np*NDT,Na*NDT*NasyncNodes),Nz);
  for(int ind=0; ind<Ntexs; ind++) {
    #ifdef USE_AIVLIB_MODEL
    //get texN from aivModel
    get_texture_size(texN[ind].x, texN[ind].y, texN[ind].z);
    #else
    // My own texN
    const int Nr=Nz/2; 
    texN[ind].x  = Np/Ns+1;
    texN[ind].y  = Na/Na+1;
    texN[ind].z  = Nr/1+1;//Nz/1+1;//Nh  ;
    tex0[ind]  = 0     ;//in_Yee_cells
    texStep[ind]  = 3.0;//in_Yee_cells
    #endif
    tex0[ind]  = 0     ;//in_Yee_cells
    texStep[ind]  = Np*3.0/(texN[ind].x-1);//in_Yee_cells

    int texNwindow = texN[ind].x;//int(ceil(Ns*NDT/texStep[ind])+2);
    #ifdef CUDA_TEX_INTERP
    texStretchHost[ind].x = 1.0/(2.0*texStep[ind]*texNwindow);
    texStretchHost[ind].y = 1.0/(2*Na*NDT)*(texN[ind].y-1)/texN[ind].y;
    texStretchHost[ind].z = 1.0/(2*Nz)*(texN[ind].z-1)/texN[ind].z;
    texStretchHost[ind].z = 1.0/(Nrmax)*(texN[ind].z-1)/texN[ind].z;
    texShiftHost[ind].x = 1.0/(2.0*texNwindow);
    texShiftHost[ind].y = 1.0/(2.0*texN[ind].y);
    texShiftHost[ind].z = 1.0/(2.0*texN[ind].z);
    #else
    texStretchHost[ind].x = 1.0/(2.0*texStep[ind]);
    texStretchHost[ind].y = 1.0/(2*Na*NDT)*(texN[ind].y-1);
    texStretchHost[ind].z = 1.0/(2*Nz)*(texN[ind].z-1);
    texStretchHost[ind].z = 1.0/(Nrmax)*(texN[ind].z-1);
    texShiftHost[ind].x = 0.5;//texN[ind].x/(2.0*texNwindow);
    texShiftHost[ind].y = 0.5;
    texShiftHost[ind].z = 0.5;
    #endif
    texsize_onhost+= texN[ind].x*texN[ind].y*texN[ind].z;
    texsize_ondevs+= texNwindow*texN[ind].y*texN[ind].z;
    if(node==0) printf("Texture%d Size %dx%dx%d (Nx x Ny x Nh)\n", ind, texN[ind].x, texN[ind].y, texN[ind].z);
    if(node==0) printf("Texture%d Stepx %g\n", ind, texStep[ind]);
    if(texStep[ind]<NDT) { printf("Texture profile step is smaller than 3*Yee_cells; Is it right?\n"); /*exit(-1);*/ }
  }
  #ifdef CUDA_TEX_INTERP
  float4 texStretchShowHost = make_float4(1.0/(2*NDT*Np)*(texN[0].x-1)/texN[0].x, 0., 0.,0.);
  float4 texShiftShowHost   = make_float4(1./(2*texN[0].x), 0., 0.,0.);
  h_scale = 2*((1<<30)/(2*texN[0].z)); const float texStretchH_host = 1.0/(texN[0].z*h_scale);
  #else
  float4 texStretchShowHost = make_float4(1.0/(2*NDT*Np)*(texN[0].x-1), 0., 0.,0.);
  float4 texShiftShowHost   = make_float4(0.5, 0., 0.,0.);
  h_scale = 2*((1<<30)/(2*texN[0].z)); const float texStretchH_host = 1.0/h_scale;
  #endif
  for(int i=0; i<NDev; i++) {
    CHECK_ERROR( cudaSetDevice(i) );
    CHECK_ERROR( cudaMemcpyToSymbol(texStretchH   ,&texStretchH_host, sizeof(float), 0, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpyToSymbol(texStretch    , texStretchHost, sizeof(float4)*Ntexs, 0, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpyToSymbol(texShift      , texShiftHost  , sizeof(float4)*Ntexs, 0, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpyToSymbol(texStretchShow, &texStretchShowHost, sizeof(float4)*Ntexs, 0, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpyToSymbol(texShiftShow  , &texShiftShowHost  , sizeof(float4)*Ntexs, 0, cudaMemcpyHostToDevice) );
  }
  CHECK_ERROR( cudaSetDevice(0) );
  if(node==0) printf("Textures data on host   : %.3fMB\n", texsize_onhost*(sizeof(coffS_t)+2*sizeof(float))/(1024.*1024.));
  if(node==0) printf("Textures data on devices: %.3fMB\n", texsize_ondevs*(sizeof(coffS_t)+2*sizeof(float))/(1024.*1024.));
  cudaChannelFormatDesc channelDesc;
  for(int ind=0; ind<Ntexs; ind++) {
    const int texNx = texN[ind].x, texNy = texN[ind].y, texNz = texN[ind].z;
    int texNwindow = texNx;//int(ceil(Ns*NDT/texStep[ind])+2);
    HostLayerS[ind] = new coffS_t[texNx*texNy*texNz]; //get pointer from aivModel
    HostLayerV[ind] = new float  [texNx*texNy*texNz]; //get pointer from aivModel
    HostLayerQ[ind] = new float2 [texNx*texNy*texNz];
    #ifndef ANISO_TR
    HostLayerT[ind]  = new float  [texNx*texNy*texNz]; //get pointer from aivModel
    #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
    HostLayerTi[ind] = new float  [texNx*texNy*texNz]; //get pointer from aivModel
    HostLayerTa[ind] = new float  [texNx*texNy*texNz]; //get pointer from aivModel
    HostLayerT[ind] = HostLayerTa[ind];
    #endif
    for(int idev=0;idev<NDev;idev++) { CHECK_ERROR( cudaSetDevice(idev) );
    printf("texNwindow=%d\n",texNwindow); 
    channelDesc = cudaCreateChannelDesc<coffS_t>(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerS[idev][ind], &channelDesc, make_cudaExtent(texNz,texNy,texNwindow)) );
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerV[idev][ind], &channelDesc, make_cudaExtent(texNz,texNy,texNwindow)) );
    channelDesc = cudaCreateChannelDesc<float2 >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerQ[idev][ind], &channelDesc, make_cudaExtent(texNz,texNy,texNwindow)) );
    #ifndef ANISO_TR
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerT [idev][ind], &channelDesc, make_cudaExtent(texNz,texNy,texNwindow)) );
    #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerTi[idev][ind], &channelDesc, make_cudaExtent(texNz,texNy,texNwindow)) );
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaMalloc3DArray(&DevLayerTa[idev][ind], &channelDesc, make_cudaExtent(texNz,texNy,texNwindow)) );
    #endif
    }
    CHECK_ERROR( cudaSetDevice(0) );
    fill_array(texStretchHost[0], texShiftHost[0]);
    ftype* rhoArr; rhoArr=new ftype[texNz+1];
    for(int ix=0; ix<texNx; ix++) for(int iy=0; iy<texNy; iy++) {
      for(int iz=0; iz<texNz; iz++) { //or get from aivModel
        // remember about yshift for idev>0
        float Vp=defCoff::Vp, Vs=defCoff::Vs, rho=defCoff::rho, drho=defCoff::drho;
        float2 Q=make_float2(0,0);
        get_mat((iz-0.5)*dz*0.5*Nrmax/(texNz-1), Vp,Vs,rho, Q);
        get_mat_tex(iz, Vp,Vs,rho, Q);
        drho=1.0/rho;
        ftype C11=Vp*Vp        , C13=Vp*Vp-2*Vs*Vs, C12=Vp*Vp-2*Vs*Vs;
        ftype C31=Vp*Vp-2*Vs*Vs, C33=Vp*Vp        , C32=Vp*Vp-2*Vs*Vs;
        ftype C21=Vp*Vp-2*Vs*Vs, C23=Vp*Vp-2*Vs*Vs, C22=Vp*Vp;
        ftype C44=Vs*Vs, C66=Vs*Vs, C55=Vs*Vs;
        #ifdef USE_AIVLIB_MODEL
        //GeoPhysPar p = get_texture_cell(ix,iy,ih-((ih==texNh)?1:0)); Vp=p.Vp; Vs=p.Vs; rho=p.sigma; drho=1.0/rho;
        GeoPhysParAniso p = get_texture_cell_aniso(ix,iy,ih-((ih==texNh)?1:0)); Vp=p.Vp; Vs=p.Vs; rho=p.sigma; drho=1.0/rho;
        if(rho==0) drho=0;

        ftype Vp_q = Vp, Vs_q1 = Vs, Vs_q2 = Vs;
        //------Anisotropy flag-------//
//        if(rho<0) { rho = -rho; Vp_q = p.Vp_q; Vs_q1 = p.Vs_q1; Vs_q2 = p.Vs_q2; }
//        ftype eps = 0, delta = 0, gamma = 0;
//        eps   = Vp_q /Vp-1;
//        delta = Vs_q1/Vs-1;
//        gamma = Vs_q2/Vs-1
        ftype eps = 0, delta = 0, gamma = 0;
        eps   = p.epsilon;
        delta = p.delta;
        gamma = p.gamma;

        ftype xx = Vp*Vp;
        ftype yy = (-Vs*Vs+sqrt((Vp*Vp-Vs*Vs)*(Vp*Vp*(1+2*delta)-Vs*Vs)));
        ftype zz = (2*eps+1)*Vp*Vp - (2*gamma+1)*2*Vs*Vs;
        ftype ww = (2*eps+1)*Vp*Vp;
        ftype ii = Vs*Vs;
        ftype aa = (2*gamma+1)*Vs*Vs;
        //C11,C12,C13;
        //C21,C22,C23;
        //C31,C32,C33;
        #else
        //if(ix<texNx/4)   Vp*= (1.0-0.5)/(texNx/4)*ix+0.5;
        //if(ix>3*texNx/4) Vp*= (0.5-1.0)/(texNx/4)*ix+0.5+4*(1.0-0.5);
        #endif
        rhoArr[iz] = rho;
        HostLayerV[ind][ix*texNy*texNz+iy*texNz+iz] = drho;
        HostLayerQ[ind][ix*texNy*texNz+iy*texNz+iz] = Q;
        #ifndef ANISO_TR
        HostLayerS[ind][ix*texNy*texNz+iy*texNz+iz] = make_float2( Vp*Vp, Vp*Vp-2*Vs*Vs )*rho;
        HostLayerT[ind][ix*texNy*texNz+iy*texNz+iz] = Vs*Vs*rho;
        #elif ANISO_TR==1
        C11 = xx; C12 = yy; C23 = zz; C22 = ww; C44 = aa; C55 = ii;
        HostLayerS [ind][ix*texNy*texNz+iy*texNz+iz] = make_float4( C11, C12, C23, C22 )*rho;
        HostLayerTa[ind][ix*texNy*texNz+iy*texNz+iz] = C44*rho;
        HostLayerTi[ind][ix*texNy*texNz+iy*texNz+iz] = C55*rho;
        #elif ANISO_TR==2
        C22 = xx; C12 = yy; C13 = zz; C11 = ww; C55 = aa; C44 = ii;
        HostLayerS [ind][ix*texNy*texNz+iy*texNz+iz] = make_float4( C22, C12, C13, C11 )*rho;
        HostLayerTa[ind][ix*texNy*texNz+iy*texNz+iz] = C55*rho;
        HostLayerTi[ind][ix*texNy*texNz+iy*texNz+iz] = C44*rho;
        #elif ANISO_TR==3
        C33 = xx; C13 = yy; C12 = zz; C11 = ww; C66 = aa; C44 = ii;
        HostLayerS [ind][ix*texNy*texNz+iy*texNz+iz] = make_float4( C33, C13, C12, C11 )*rho;
        HostLayerTa[ind][ix*texNy*texNz+iy*texNz+iz] = C66*rho;
        HostLayerTi[ind][ix*texNy*texNz+iy*texNz+iz] = C44*rho;
        #else
        #error ANISO_TYPE ANISO_TR not implemented yet
        #endif
      }
      #ifdef USE_AIVLIB_MODEL
      if(iy==0) { printf("Testing get_h ix=%d/%d \r", ix, texNx-1); fflush(stdout); }
      int aivTexStepX=Np*NDT*2/(texNx-1); //in half-YeeCells
      int aivTexStepY=2*Nz/(texNy-1); //in half-YeeCells
      for(int xx=(ix==texNx-1?1:0); xx<((ix==0)?1:aivTexStepX); xx++) for(int yy=(iy==texNy-1?1:0); yy<((iy==0)?1:aivTexStepY); yy++) {
        for(int iz=0; iz<Na*NDT*2; iz++) {
          unsigned short h = get_h(ix*aivTexStepX-xx, iy*aivTexStepY-yy, min(0.,Npmly/2*NDT-iz*0.5)*da);
          int id = h/(2*h_scale), idd=h%(2*h_scale); 
          //int id = floor((h)/double(1<<16)*112);
          float rho1 = rhoArr[2*id];
          float rho2 = rhoArr[2*id+1];
          if(id<0 || 2*id>=texNh || idd>h_scale || rho1<=0 || rho2<=0)
             printf("Error: ix=%d-%d iy=%d-%d iz=%g id=%d h%%h_scale=%d rho1=%g rho2=%g\n", ix*aivTexStepX, xx, iy*aivTexStepY, yy, -iz*0.5*da, id, idd, rho1,rho2);
        }
      }
      #endif
    }
    delete rhoArr;
  }
  printf("\n");

  for(int idev=0;idev<NDev;idev++) { CHECK_ERROR( cudaSetDevice(idev) ); 
  CHECK_ERROR( cudaMemcpy(layerS [idev], layerS_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerV [idev], layerV_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerQ [idev], layerQ_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerT [idev], layerT_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerTi[idev], layerTi_host[idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  CHECK_ERROR( cudaMemcpy(layerTa[idev], layerTa_host[idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  }
  CHECK_ERROR( cudaSetDevice(0) );

  if(node==0) printf("creating texture objects...\n");
  for(int idev=0;idev<NDev;idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    #ifdef USE_TEX_REFS
    layerRefS.addressMode[0] = cudaAddressModeClamp; layerRefV.addressMode[0] = cudaAddressModeClamp; layerRefQ.addressMode[0] = cudaAddressModeClamp; layerRefT.addressMode[0] = cudaAddressModeClamp; layerRefTi.addressMode[0] = cudaAddressModeClamp; layerRefTa.addressMode[0] = cudaAddressModeClamp;
    layerRefS.addressMode[1] = cudaAddressModeClamp; layerRefV.addressMode[1] = cudaAddressModeClamp; layerRefQ.addressMode[1] = cudaAddressModeClamp; layerRefT.addressMode[1] = cudaAddressModeClamp; layerRefTi.addressMode[1] = cudaAddressModeClamp; layerRefTa.addressMode[1] = cudaAddressModeClamp;
    layerRefS.addressMode[2] = cudaAddressModeWrap;  layerRefV.addressMode[2] = cudaAddressModeWrap;  layerRefQ.addressMode[2] = cudaAddressModeWrap;  layerRefT.addressMode[2] = cudaAddressModeWrap;  layerRefTi.addressMode[2] = cudaAddressModeWrap;  layerRefTa.addressMode[2] = cudaAddressModeWrap;
    layerRefS.filterMode = cudaFilterModeLinear; layerRefV.filterMode = cudaFilterModeLinear; layerRefQ.filterMode = cudaFilterModeLinear; layerRefT.filterMode = cudaFilterModeLinear;layerRefTi.filterMode = cudaFilterModeLinear;layerRefTa.filterMode = cudaFilterModeLinear;
    #ifdef CUDA_TEX_INTERP
    layerRefS.normalized = true; layerRefV.normalized = true; layerRefQ.normalized = true; layerRefT.normalized = true; layerRefTi.normalized = true; layerRefTa.normalized = true;
    #else                                                     
    layerRefS.normalized =false; layerRefV.normalized =false; layerRefQ.normalized =false; layerRefT.normalized =false; layerRefTi.normalized =false; layerRefTa.normalized =false;
    #endif
    channelDesc = cudaCreateChannelDesc<coffS_t>(); CHECK_ERROR( cudaBindTextureToArray(layerRefS , DevLayerS [idev][0], channelDesc) );
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(layerRefV , DevLayerV [idev][0], channelDesc) );
    channelDesc = cudaCreateChannelDesc<float2 >(); CHECK_ERROR( cudaBindTextureToArray(layerRefQ , DevLayerQ [idev][0], channelDesc) );
    #ifndef ANISO_TR
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(layerRefT , DevLayerT [idev][0], channelDesc) );
    #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(layerRefTi, DevLayerTi[idev][0], channelDesc) );
    channelDesc = cudaCreateChannelDesc<float  >(); CHECK_ERROR( cudaBindTextureToArray(layerRefTa, DevLayerTa[idev][0], channelDesc) );
    #endif//ANISO_TR
    #endif//USE_TEX_REFS

    CHECK_ERROR( cudaMemcpy(layerS [idev], layerS_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerV [idev], layerV_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerQ [idev], layerQ_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerT [idev], layerT_host [idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerTi[idev], layerTi_host[idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
    CHECK_ERROR( cudaMemcpy(layerTa[idev], layerTa_host[idev], sizeof(cudaTextureObject_t)*Ntexs, cudaMemcpyHostToDevice) );
  }
  CHECK_ERROR(cudaSetDevice(0));
  copyAllTexs();
  
  //-------------Volfrac and SurfFrac Textures--------------------------//
  /*
  const int texFnx=1;
  const int texFny=1;
  const int texRd=1;
  float2* VolFracHost = new float2[texFnx*texFny*texRd];
  float2* SurFracHost = new float2[texFnx*texFny*texRd];
  cudaArray* VolFracDev[NDev];
  cudaArray* SurFracDev[NDev];
  //volfractex
  //surfractex
  //cudaChannelFormatDesc channelDesc;
  for(int idev=0;idev<NDev;idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    channelDesc = cudaCreateChannelDesc<float2>(); CHECK_ERROR( cudaMalloc3DArray(&VolFracDev[idev], &channelDesc, make_cudaExtent(texRd,texFny,texFnx)) );
    channelDesc = cudaCreateChannelDesc<float2>(); CHECK_ERROR( cudaMalloc3DArray(&SurFracDev[idev], &channelDesc, make_cudaExtent(texRd,texFny,texFnx)) );
    fill_vol_surf_frac_array(VolFracHost, SurFracHost, texFnx, texFny, texRd);
    volfractex.addressMode[0] = cudaAddressModeClamp; surfractex.addressMode[0] = cudaAddressModeClamp;
    volfractex.addressMode[1] = cudaAddressModeClamp; surfractex.addressMode[1] = cudaAddressModeClamp;
    volfractex.addressMode[2] = cudaAddressModeClamp; surfractex.addressMode[2] = cudaAddressModeClamp;
    volfractex.filterMode = cudaFilterModeLinear    ; surfractex.filterMode = cudaFilterModeLinear;
    volfractex.normalized = true                    ; surfractex.normalized = true;
    channelDesc = cudaCreateChannelDesc<float2  >(); CHECK_ERROR( cudaBindTextureToArray(volfractex, VolFracDev[idev], channelDesc) );
    channelDesc = cudaCreateChannelDesc<float2  >(); CHECK_ERROR( cudaBindTextureToArray(surfractex, SurFracDev[idev], channelDesc) );

    cudaMemcpy3DParms copyparms={0}; copyparms.srcPos=make_cudaPos(0,0,0); copyparms.dstPos=make_cudaPos(0,0,0);
    copyparms.kind=cudaMemcpyHostToDevice;
    copyparms.srcPtr = make_cudaPitchedPtr(&VolFracHost[0], texRd*sizeof(float2), texFny, texFnx);
    copyparms.dstArray = VolFracDev[idev];
    copyparms.extent = make_cudaExtent(texFnx,texFny,texRd);
    CHECK_ERROR( cudaMemcpy3D(&copyparms) );
    copyparms.srcPtr = make_cudaPitchedPtr(&SurFracHost[0], texRd*sizeof(float2), texFny, texFnx);
    copyparms.dstArray = SurFracDev[idev];
    copyparms.extent = make_cudaExtent(texFnx,texFny,texRd);
    CHECK_ERROR( cudaMemcpy3D(&copyparms) );
  }
  delete[] VolFracHost, SurFracHost;*/
}

void ModelTexs::copyTexs(const int x1dev, const int x2dev, const int x1host, const int x2host, cudaStream_t streamCopy[NDev]){
}
void ModelTexs::copyAllTexs(){
  for(int idev=0;idev<NDev;idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    for(int ind=0; ind<Ntexs; ind++) {
      int numX = texN[ind].x;

      const int texNy = texN[ind].y, texNz = texN[ind].z;
      cudaMemcpy3DParms copyparms={0}; copyparms.srcPos=make_cudaPos(0,0,0); copyparms.dstPos=make_cudaPos(0,0,0);
      copyparms.kind=cudaMemcpyHostToDevice;
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerS[ind][0], texNz*sizeof(coffS_t), texNz, texNy);
      copyparms.dstArray = DevLayerS[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerV[ind][0], texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerV[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerQ[ind][0], texNz*sizeof(float2 ), texNz, texNy);
      copyparms.dstArray = DevLayerQ[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );
      #ifndef ANISO_TR
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerT[ind][0], texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerT[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );
      #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerTi[ind][0],texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerTi[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerTa[ind][0],texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerTa[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3D(&copyparms) );
      #else
      #error UNKNOWN ANISO_TYPE
      #endif
    }
  }
  CHECK_ERROR(cudaSetDevice(0));
}

void ModelTexs::copyTexs(const int xdev, const int xhost, cudaStream_t streamCopy[NDev]){ 
  return;
  if(xhost==Np) for(int ind=0; ind<Ntexs; ind++) copyTexs(xhost+ceil(texStep[ind]/NDT), xhost+ceil(texStep[ind]/NDT), streamCopy);
  for(int idev=0;idev<NDev;idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    for(int ind=0; ind<Ntexs; ind++) {
      int texNwindow = int(ceil(Ns*NDT/texStep[ind])+2);
      //if(xhost*NDT<=tex0[ind] || xhost*NDT>tex0[ind]+texN[ind].x*texStep[ind]) continue;
      if(xhost*NDT<=tex0[ind]) continue;
      if(floor(xhost*NDT/texStep[ind])==floor((xhost-1)*NDT/texStep[ind])) continue;
      int storeX  = int(floor(xhost*NDT/texStep[ind])-1+texNwindow)%texNwindow;
      int loadX = int(floor((xhost*NDT-tex0[ind])/texStep[ind])-1);
      double numXf = NDT/texStep[ind];
      int numX = (numXf<=1.0)?1:floor(numXf);
      //while(storeX+numX>texNwindow) numX--;

      DEBUG_PRINT(("copy Textures to dev%d, ind=%d hostx=%d -> %d=devx (num=%d) // texNwindow=%d\n", idev, ind, loadX, storeX, numX, texNwindow));

      const int texNy = texN[ind].y, texNz = texN[ind].z;
      cudaMemcpy3DParms copyparms={0}; copyparms.srcPos=make_cudaPos(0,0,loadX); copyparms.dstPos=make_cudaPos(0,0,storeX);
      copyparms.kind=cudaMemcpyHostToDevice;
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerS[ind][0], texNz*sizeof(coffS_t), texNz, texNy);
      copyparms.dstArray = DevLayerS[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy[idev]) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerV[ind][0], texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerV[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy[idev]) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerQ[ind][0], texNz*sizeof(float2 ), texNz, texNy);
      copyparms.dstArray = DevLayerQ[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy[idev]) );
      #ifndef ANISO_TR
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerT[ind][0], texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerT[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy[idev]) );
      #elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerTi[ind][0],texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerTi[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy[idev]) );
      copyparms.srcPtr = make_cudaPitchedPtr(&HostLayerTa[ind][0],texNz*sizeof(float  ), texNz, texNy);
      copyparms.dstArray = DevLayerTa[idev][ind];
      copyparms.extent = make_cudaExtent(texNz,texNy,numX);
      CHECK_ERROR( cudaMemcpy3DAsync(&copyparms, streamCopy[idev]) );
      #else
      #error UNKNOWN ANISO_TYPE
      #endif
    }
  }
  CHECK_ERROR(cudaSetDevice(0));
}

#include "cuda_fp16.h"
inline ftype4 get_averaged(double x, double y, double z, int& split, double drx=dx, double dry=dy, double drz=dz, int parall=0){
  float4 coffCnt = get_mat(x,y,z);
  split=0;
  const double cutoff = 1e-3;//*8*8;
  const double Vol = drx*dry*drz; const double Vol0 = dx*dy*dz;
  if(Vol<=Vol0*cutoff) return coffCnt*Vol;
  float4 coffCrn[8]; for(int i=0;i<8;i++) coffCrn[i]=get_mat(x-drx*0.5+drx*(i>>0&1), y-dry*0.5+dry*(i>>1&1), z-drz*0.5+drz*(i>>2&1));
  for(int i=0;i<8;i++) if(coffCrn[i].x!=coffCnt.x || coffCrn[i].y!=coffCnt.y || coffCrn[i].z!=coffCnt.z || coffCrn[i].w!=coffCnt.w) { split=1; break; }
  if(split==0) return coffCnt*Vol;
  else {
    ftype4 retC=make_float4(0,0,0,0); int aaaaa=0;
    if(parall==1) {
      #pragma omp parallel for
      for(int i=0;i<8;i++) retC+= get_averaged(x-drx*0.25+drx*0.5*(i>>0&1), y-dry*0.25+dry*0.5*(i>>1&1), z-drz*0.25+drz*0.5*(i>>2&1), aaaaa, 0.5*drx, 0.5*dry, 0.5*drz);
    }
    else {
      for(int i=0;i<8;i++) retC+= get_averaged(x-drx*0.25+drx*0.5*(i>>0&1), y-dry*0.25+dry*0.5*(i>>1&1), z-drz*0.25+drz*0.5*(i>>2&1), aaaaa, 0.5*drx, 0.5*dry, 0.5*drz);
    }
    return retC;
  }
}
inline float2 calc_surf(double x, double y, double z, double dx1, double dx2, int dir, double Area0){
  int2 matCnt = get_sphere(x,y,z);
  const double cutoff = 1e-2;//*8*8;
  const double drr0[3] = {dx ,dy ,dz };
  double Area =dx1*dx2;
  //double Area0=1; for(int i=0; i<3; i++) if(i!=dir) Area0*drr0[i];
  if(Area<=Area0*cutoff) return make_float2(matCnt)*Area;
  int2 matCrn[4];
  for(int i=0;i<4;i++) {
    double cc[3] = {x,y,z};
    cc[(dir+1)%3]+= -dx1*0.5+dx1*(i>>0&1);
    cc[(dir+2)%3]+= -dx2*0.5+dx2*(i>>1&1);
    matCrn[i] = get_sphere(cc[0], cc[1], cc[2]);
  }
  int split=0;
  for(int i=0;i<4;i++) if(matCrn[i].x!=matCnt.x || matCrn[i].y!=matCnt.y) 
  split=1;
  if(split==0) return make_float2(matCnt)*Area;
  else {
    float2 retC=make_float2(0,0);
    for(int i=0;i<4;i++) {
      double cc[3] = {x,y,z};
      cc[(dir+1)%3]+= -dx1*0.25+dx1*0.5*(i>>0&1);
      cc[(dir+2)%3]+= -dx2*0.25+dx2*0.5*(i>>1&1);
      retC+= calc_surf(cc[0],cc[1],cc[2], 0.5*dx1,0.5*dx2, dir, Area0);
    }
    return retC;
  }
}
inline float2 calc_vol(double x, double y, double z, double drx=dx, double dry=dy, double drz=dz, int parall=0){
  int2 coffCnt = get_sphere(x,y,z);
  int split = 0;
  const double cutoff = 1e-3;//*8*8;
  const double Vol = drx*dry*drz; const double Vol0 = dx*dy*dz;
  if(Vol<=Vol0*cutoff) return make_float2(coffCnt)*Vol;
  int2 coffCrn[8]; for(int i=0;i<8;i++) coffCrn[i]=get_sphere(x-drx*0.5+drx*(i>>0&1), y-dry*0.5+dry*(i>>1&1), z-drz*0.5+drz*(i>>2&1));
  for(int i=0;i<8;i++) if(coffCrn[i].x!=coffCnt.x || coffCrn[i].y!=coffCnt.y) { split=1; break; }
  if(split==0) return make_float2(coffCnt)*Vol;
  else {
    float2 retC=make_float2(0,0);
    if(parall==1) {
      #pragma omp parallel for
      for(int i=0;i<8;i++) retC+= calc_vol(x-drx*0.25+drx*0.5*(i>>0&1), y-dry*0.25+dry*0.5*(i>>1&1), z-drz*0.25+drz*0.5*(i>>2&1), 0.5*drx, 0.5*dry, 0.5*drz);
    }
    else {
      for(int i=0;i<8;i++) retC+= calc_vol(x-drx*0.25+drx*0.5*(i>>0&1), y-dry*0.25+dry*0.5*(i>>1&1), z-drz*0.25+drz*0.5*(i>>2&1), 0.5*drx, 0.5*dry, 0.5*drz);
    }
    return retC;
  }
}
float my_get_q(int x, int y, int z){
  return 0;
  /*double px=x*dx*0.5; double py=y*dy*0.5; double pz=z*dz*0.5;
  float3 cellcnt = make_ftype3(px,py,pz);
  const float3 Earth_Center = make_ftype3(EARTH_Center_X*dx, EARTH_Center_Y*dy, EARTH_Center_Z*dz);
  double r = length(cellcnt-Earth_Center);
  //r = fabs(cellcnt.y-Earth_Center.y);
  ftype Q=0;
  
  if(r>=6600-1) r=6600-1;
  Q = interpolate_ubs<3>(Qt , r/40-floor(r/40), int(r/40));
  return (2-Q*dt)/(2+Q*dt);*/
}
long my_get_h(int x, int y, int z){
  double ss=1; int split=0;
  //float4 coff_aver = get_averaged(x*dx*0.5, y*dy*0.5, z*dz*0.5, split, dx*ss,dy*ss,dz*ss);
  //coff_aver/=(dx*dy*dz*ss*ss*ss);
  float4 coff  = get_mat(x*dx*0.5, y*dy*0.5, z*dz*0.5);
  //float4 coff_aver = coff;
  double Tcoff = double(coff.w);
  double Vcoff = double(coff.x);//double(1/(defCoff::rho*coff_aver.x*defCoff::rho)); if(Vcoff>2) Vcoff=2;
  //double Vcoff = double(coff.x);
  float2 Scoff = make_float2(coff.y,coff.z);
  long h=0;

      /*float2 coff_aver_small = calc_vol(x*dx*0.5, y*dy*0.5, z*dz*0.5, dx,dy,dz,1);
      coff_aver_small/=(dx*dy*dz);
      
      if(coff_aver_small.x==0) {Tcoff=0; Scoff.x=0; Scoff.y=0;}
      if(coff_aver_small.y==0) {Tcoff=0; Scoff.y=Scoff.x; }*/
      //else if(coff_aver_small<1-1e-10) {Tcoff*=-1; Scoff.x*=-1; Scoff.y*=-1;}

  double q_coff = my_get_q(x,y,z);
  Tcoff*=(1+q_coff)/2; 
  Scoff*=(1+q_coff)/2; 
  if((x%2)+(y%2)+(z%2)==0) return 0;
  if((x%2)+(y%2)+(z%2)==1) h = *((long*)&Tcoff); // T
  if((x%2)+(y%2)+(z%2)==2) h = *((long*)&Vcoff); // V
  if((x%2)+(y%2)+(z%2)==3) h = *((long*)&Scoff); // S
  /*if((y%2)==0 && split==1) { return 0; }*/
  return h;
}

float2 get_s(int x, int y, int z, int dir, ftype drx1, ftype drx2){
  return calc_surf(x*0.5*dx,y*0.5*dy,z*0.5*dz, drx1, drx2, dir, drx1*drx2)/(drx1*drx2);
};
void ModelRag::set(int x, int y) {
  return;
/*
    #if TEX_MODEL_TYPE==1
    for(int i=0;i<4        ;i++) for(int iz=0;iz<Nz;iz++) I[i][iz]=0;
    #endif
    for(int i=0;i<32;i++) for(int iz=0;iz<Nz;iz++) { h[i][iz].x=0; h[i][iz].y=0; }
  // set values from aivModel
  // remember about yshift for idev>0
    int idev=0; int ym=0;
    while(y>=ym && idev<NDev) { ym+=NStripe[idev]; idev++; }
    y-= idev-1;
    const int d_index[64][3] = { {-3, +3, 1}, {-2, +3, 0}, {-2, +4, 1}, {-1, +4, 0}, {-1, +5, 1}, {+0, +5, 0}, 
                                 {-2, +2, 1}, {-1, +2, 0}, {-1, +3, 1}, {+0, +3, 0}, {+0, +4, 1}, {+1, +4, 0}, 
                                 {-1, +1, 1}, {+0, +1, 0}, {+0, +2, 1}, {+1, +2, 0}, {+1, +3, 1}, {+2, +3, 0}, 
                                 {+0, +0, 1}, {+1, +0, 0}, {+1, +1, 1}, {+2, +1, 0}, {+2, +2, 1}, {+3, +2, 0},
                                 {+1, -1, 1}, {+2, -1, 0}, {+2, +0, 1}, {+3, +0, 0}, {+3, +1, 1}, {+4, +1, 0}, 
                                 {+2, -2, 1}, {+3, -2, 0}, {+3, -1, 1}, {+4, -1, 0}, {+4, +0, 1}, {+5, +0, 0},

                                 {-3, +0, 1}, {-2, -1, 1}, {-1, -1, 0}, {-1, -2, 1}, 
                                 {-2, +1, 1}, {-1, +1, 0}, {-1, +0, 1}, {+0, -1, 1}, {+1, -1, 0}, 
                                 {-1, +2, 1}, {+0, +1, 1}, {+1, +1, 0}, {+1, +0, 1}, 
                                 {+0, +3, 1}, {+1, +3, 0}, {+1, +2, 1}, {+2, +1, 1}, {+3, +1, 0}, 
                                 {+1, +4, 1}, {+2, +3, 1}, {+3, +3, 0}, {+3, +2, 1}, 
                                 {+2, +5, 1}, {+3, +5, 0}, {+3, +4, 1}, {+4, +3, 1}, {+5, +3, 0},
                                 {0,0,0} };
    #ifdef USE_AIVLIB_MODEL
    const double corrCoff1 = 1.0/double(H_MAX_SIZE)*(parsHost.texs.texN[0].z-1);
    const double corrCoff2 = 1.0/parsHost.texs.texN[0].z*H_MAX_SIZE;
    for(int i=0;i<32;i++) for(int iz=0;iz<Nz;iz++) {
      int3 x4h;
      x4h = make_int3(x*2*NDT+d_index[2*i  ][0], iz*2+d_index[2*i  ][2], y*2*NDT+d_index[2*i  ][1]); x4h = check_bounds(x4h);
      h[i][iz].x = get_h(x4h.x, x4h.y, min(0.,Npmly/2*NDT-x4h.z*0.5)*dy) + parsHost.texs.h_scale/2;
      //h[i][iz].x = ((x4h.x*x4h.y-x4h.z*0.5*dy)*corrCoff1+0.5)*corrCoff2;
      x4h = make_int3(x*2*NDT+d_index[2*i+1][0], iz*2+d_index[2*i+1][2], y*2*NDT+d_index[2*i+1][1]); x4h = check_bounds(x4h);
      h[i][iz].y = get_h(x4h.x, x4h.y, min(0.,Npmly/2*NDT-x4h.z*0.5)*dy) + parsHost.texs.h_scale/2;
      //h[i][iz].y = ((x4h.x*x4h.y-x4h.z*0.5*dy)*corrCoff1+0.5)*corrCoff2;
    }
    #else
    for(int i=0;i<32;i++) for(int iz=0;iz<Nz;iz++) {
      int3 x4h;
      x4h = make_int3(x*2*NDT+d_index[2*i  ][0], y*2*NDT+d_index[2*i  ][1], iz*2+d_index[2*i  ][2]);
      h[i][iz].x = my_get_h(x4h.x, x4h.y, x4h.z);
      q[i][iz].x = my_get_q(x4h.x, x4h.y, x4h.z);

      x4h = make_int3(x*2*NDT+d_index[2*i+1][0], y*2*NDT+d_index[2*i+1][1], iz*2+d_index[2*i+1][2]);
      h[i][iz].y = my_get_h(x4h.x, x4h.y, x4h.z);
      q[i][iz].y = my_get_q(x4h.x, x4h.y, x4h.z);
    }
    const int df_index[4][18][3] = { { {-3, +3, 1}, {-2, +3, 0}, {-2, +4, 1}, {-1, +4, 0}, {-1, +5, 1}, {+0, +5, 0}, 
                                       {-2, +2, 1}, {-1, +2, 0}, {-1, +3, 1}, {+0, +3, 0}, {+0, +4, 1}, {+1, +4, 0}, 
                                       {-1, +1, 1}, {+0, +1, 0}, {+0, +2, 1}, {+1, +2, 0}, {+1, +3, 1}, {+2, +3, 0} },

                                     { {+0, +0, 1}, {+1, +0, 0}, {+1, +1, 1}, {+2, +1, 0}, {+2, +2, 1}, {+3, +2, 0},
                                       {+1, -1, 1}, {+2, -1, 0}, {+2, +0, 1}, {+3, +0, 0}, {+3, +1, 1}, {+4, +1, 0}, 
                                       {+2, -2, 1}, {+3, -2, 0}, {+3, -1, 1}, {+4, -1, 0}, {+4, +0, 1}, {+5, +0, 0} },

                                     { {-3, +0, 1}, {-2, +0, 0}, {-2, -1, 1}, {-1, -1, 0}, {-1, -2, 1}, {+0, -2, 0},
                                       {-2, +1, 1}, {-1, +1, 0}, {-1, +0, 1}, {+0, +0, 0}, {+0, -1, 1}, {+1, -1, 0}, 
                                       {-1, +2, 1}, {+0, +2, 0}, {+0, +1, 1}, {+1, +1, 0}, {+1, +0, 1}, {+2, +0, 0} },

                                     { {+0, +3, 1}, {+1, +3, 0}, {+1, +2, 1}, {+2, +2, 0}, {+2, +1, 1}, {+3, +1, 0}, 
                                       {+1, +4, 1}, {+2, +4, 0}, {+2, +3, 1}, {+3, +3, 0}, {+3, +2, 1}, {+4, +2, 0},
                                       {+2, +5, 1}, {+3, +5, 0}, {+3, +4, 1}, {+4, +4, 0}, {+4, +3, 1}, {+5, +3, 0} } };
    for(int idmd=0; idmd<4; idmd++) {
      for(int iz=0;iz<Nz;iz++) {
        int into=0; int out=0;
        sInd[idmd][iz]=0;
        for(int i=0;i<18;i++) {
          int3 crd = make_int3(x*2*NDT+df_index[idmd][i][0], y*2*NDT+df_index[idmd][i][1], iz*2+df_index[idmd][i][2]);
          ftype3 cellcnt = make_ftype3(crd.x*dx*0.5, crd.y*dy*0.5, crd.z*dz*0.5);
          const int PStype=0;
          const ftype r = get_radius(cellcnt.x,cellcnt.y,cellcnt.z, PStype);
          const ftype3 Earth_Center = make_ftype3(EARTH_Center_X*dx, EARTH_Center_Y*dy, EARTH_Center_Z*dz);
          const ftype delta = length(cellcnt-Earth_Center)-r;
          if(fabs(delta)<3*dx) { sInd[idmd][iz]=1; break; }
          else if(delta<0) into=1;
          else if(delta>=0) out=1;
        }
        if(sInd[idmd][iz]==0 && out && !into) sInd[idmd][iz]=2;
      }
    }
    #endif
    */
}


