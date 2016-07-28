#ifndef TEXMODEL_CU
#define TEXMODEL_CU
#include <stdio.h>
#include "cuda_math.h"
#ifdef USE_AIVLIB_MODEL
#include "spacemodel/include/access2model.hpp"
#endif//USE_AIVLIB_MODEL

#define USE_TEX_REFS
#define MAX_TEXS 10

//1 -- many textures, 2 -- one texture, 0 -- one texture reference
#define TEX_MODEL_TYPE 0
//#define H_MAX_SIZE (USHRT_MAX+1)
//#define H_MAX_SIZE (INT_MAX+1)
#define H_MAX_SIZE (1<<30)
typedef long2 htype;

//const float texStretchX=1.0/(2*Ns*NDT);
//const float texStretchY=1.0/(2*Nz);
extern __constant__ float texStretchH;
extern __constant__ float4 texStretch[MAX_TEXS];
extern __constant__ float4 texShift[MAX_TEXS];
extern __constant__ float4 texStretchShow;
extern __constant__ float4 texShiftShow;
struct Surfs{ ftype3 s[4]; };
struct __align__(16) ModelRag{
  #if TEX_MODEL_TYPE==1
  int I[4][Nz];
  #endif
  //-----------int sInd[4][Nz];
  //htype h[NDT*NDT*7+1][Nz];
  //-----------htype h[32][Nz];
  //-----------float2 q[32][Nz];
  //Surfs s[27][Nz];
  int3 check_bounds(const int3 &v) {
    int3 ret=v;
    if(v.x<0) ret.x=0; else if(v.x>=Np*NDT*2) ret.x=Np*NDT*2-1;
    if(v.y<0) ret.y=0; else if(v.y>=Nz*2    ) ret.y=Nz*2-1;
    if(v.z<0) ret.z=0; else if(v.z>=Na*NDT*2) ret.z=Na*NDT*2-1;
    return ret;
  }
  void set(int x, int y);
};

#ifndef ANISO_TR
typedef float2 coffS_t;
typedef ftype2 coffS_tp;
#define DEF_COFF_S make_ftype2(defCoff::Vp*defCoff::Vp*defCoff::rho*dtdrd24, (defCoff::Vp*defCoff::Vp-2*defCoff::Vs*defCoff::Vs)*defCoff::rho*dtdrd24);
#elif ANISO_TR==1 || ANISO_TR==2 || ANISO_TR==3
typedef float4 coffS_t;
typedef ftype4 coffS_tp;
#else // ifndef ANISO_TR
#error UNKNOWN ANISOTROPY TYPE
#endif //ANISO_TR
#ifdef USE_TEX_REFS
extern texture<coffS_t, cudaTextureType3D, cudaReadModeElementType> layerRefS;
extern texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefV;
extern texture<float2 , cudaTextureType3D, cudaReadModeElementType> layerRefQ;
extern texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefT;
extern texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefTa;
extern texture<float  , cudaTextureType3D, cudaReadModeElementType> layerRefTi;
#endif//USE_TEX_REFS
extern texture<coffS_t, cudaTextureType3D, cudaReadModeElementType> radTexS;
extern texture<float  , cudaTextureType3D, cudaReadModeElementType> radTexV;
extern texture<float  , cudaTextureType3D, cudaReadModeElementType> radTexT;
extern texture<float2 , cudaTextureType3D, cudaReadModeElementType> radTexQ;
struct ModelTexs{
  bool ShowTexBinded;
  int Ntexs;
  int h_scale;
  int3* texN;
  int* tex0;
  float* texStep;
  //cudaTextureObject_t layerS[MAX_TEXS], layerV[MAX_TEXS], layerT[MAX_TEXS];
  cudaTextureObject_t TexlayerS[NDev], TexlayerV[NDev], TexlayerQ[NDev], TexlayerT[NDev], TexlayerTi[NDev], TexlayerTa[NDev];
  cudaTextureObject_t *layerS[NDev], *layerV[NDev], *layerQ[NDev], *layerT[NDev], *layerTi[NDev], *layerTa[NDev];
  cudaTextureObject_t *layerS_host[NDev], *layerV_host[NDev], *layerQ_host[NDev], *layerT_host[NDev], *layerTi_host[NDev], *layerTa_host[NDev];
  coffS_t** HostLayerS; float **HostLayerV, **HostLayerT, **HostLayerTi, **HostLayerTa; float2 **HostLayerQ;
  cudaArray** DevLayerS[NDev], **DevLayerV[NDev], **DevLayerQ[NDev], **DevLayerT[NDev], **DevLayerTi[NDev], **DevLayerTa[NDev];
  
  ftype* Vp, *Vs, *Rho; ftype2 *Qg;
  void init();
  void copyTexs(const int x1dev, const int x2dev, const int x1host, const int x2host, cudaStream_t streamCopy[NDev]);
  void copyTexs(const int xdev, const int xhost, cudaStream_t streamCopy[NDev]);
  void copyAllTexs();
  void fill_array(float4, float4);
};

#include "signal.h"
namespace defCoff {
//  const ftype Vp=TFSF::Vp_, Vs=TFSF::Vs_, rho=TFSF::Rho,drho=TFSF::dRho;
  const ftype Vp=5.8, Vs=3.2, rho=2.6,drho=1/rho;
  const ftype C11 = Vp*Vp        , C12 = Vp*Vp-2*Vs*Vs, C13 = Vp*Vp-2*Vs*Vs;
  const ftype C21 = Vp*Vp-2*Vs*Vs, C22 = Vp*Vp        , C23 = Vp*Vp-2*Vs*Vs;
  const ftype C31 = Vp*Vp-2*Vs*Vs, C32 = Vp*Vp-2*Vs*Vs, C33 = Vp*Vp;
};
#if ANISO_TR==1
#define DEF_COFF_S make_ftype4(defCoff::C11, defCoff::C12, defCoff::C23, defCoff::C22)*defCoff::rho*dtdrd24;
#elif ANISO_TR==2
#define DEF_COFF_S make_ftype4(defCoff::C22, defCoff::C12, defCoff::C13, defCoff::C11)*defCoff::rho*dtdrd24;
#elif ANISO_TR==3
#define DEF_COFF_S make_ftype4(defCoff::C33, defCoff::C13, defCoff::C12, defCoff::C11)*defCoff::rho*dtdrd24;
#endif//ANISO_TR

extern texture <float2  , cudaTextureType3D, cudaReadModeElementType> volfractex;
extern texture <float2  , cudaTextureType3D, cudaReadModeElementType> surfractex;

#endif//TEXMODEL_CU
