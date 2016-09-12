#include <stdio.h>
#include <errno.h>
#include <omp.h>
#include <semaphore.h>
#ifdef MPI_ON
#include <mpi.h>
#endif
#include "chooseV.h"
#include "signal.h"
#ifdef MPI_ON
MPI_Datatype MPI_DMDRAGTYPE;
MPI_Datatype MPI_RAGPMLTYPE;
MPI_Datatype MPI_HLFRAGTYPE;
#endif

int* mapNodeSize;
//=============================================
ftype* __restrict__ hostKpmlx1; ftype* __restrict__ hostKpmlx2;
ftype* __restrict__ hostKpmly1; ftype* __restrict__ hostKpmly2;
ftype* __restrict__ hostKpmlz1; ftype* __restrict__ hostKpmlz2;
GeoParamsHost parsHost;
__constant__ GeoParams pars;
__constant__ int devNStripe[NDev] = STRIPES;
__constant__ ftype Kpmlx1[(KNpmlx==0)?1:KNpmlx];
__constant__ ftype Kpmlx2[(KNpmlx==0)?1:KNpmlx];
__constant__ ftype Kpmly1[(KNpmly==0)?1:KNpmly];
__constant__ ftype Kpmly2[(KNpmly==0)?1:KNpmly];
__constant__ ftype Kpmlz1[(KNpmlz==0)?1:KNpmlz];
__constant__ ftype Kpmlz2[(KNpmlz==0)?1:KNpmlz];
//__shared__ ftype2 shared_fld[2][7][Nz];
//__shared__ ftype2 shared_fld[(FTYPESIZE*Nv*28>0xc000)?7:14][Nv];
__shared__ ftype2 shared_fld[SHARED_SIZE][NzMax];
texture<char, cudaTextureType3D> index_tex;
cudaArray* index_texArray=0;

#include "window.hpp"
struct AsyncMPIexch{
  int even,ix,t0,Nt,mpirank; bool do_run;
  double exch_time;
  sem_t sem_mpi, sem_calc;
  void exch(const int _even, const int _ix, const int _t0, const int _Nt, const int _mpirank) {
    even=_even; ix=_ix; t0=_t0; Nt=_Nt, mpirank=_mpirank;
    exch_time=0;
    if(sem_post(&sem_mpi)<0) printf("exch sem_post error %d\n",errno);
  }
  void exch_sync(){ if(sem_wait(&sem_calc)<0) printf("exch_sync sem error %d\n",errno); }
  void run() {
    if(sem_wait(&sem_mpi)<0) printf("run sem_wait error %d\n",errno);
    if(do_run==0) return;
    double start_time = omp_get_wtime();
    if(even==0) DiamondRag::bufSendMPIp(mpirank, t0,Nt);
    if(even==1) DiamondRag::bufSendMPIm(mpirank, t0,Nt);
    exch_time = omp_get_wtime()-start_time;
    if(sem_post(&sem_calc)<0) printf("run sem_post error %d\n",errno);;
  }
} ampi_exch;
#ifdef TIMERS_ON
#define IFPMLS(func,a,b,c,d,TIMER,args) {\
  /*printf(#func" idev=%d ix=%d iym=%d Nblocks=%d\n", idev,ix, iym, a);*/ TIMER.init(d); \
  if(isPMLs) PMLS##func<<<a,b,c,d>>>args; else func<<<a,b,c,d>>>args; TIMER.record(); }
#else
#define IFPMLS(func,a,b,c,d,EVENT,args) {\
  /*printf(#func" PMLS=%d idev=%d w0=%d ix=%d iym=%d Nblocks=%d\n", isPMLs, idev,w0,ix, iym, a);*/\
  for(int iz=0   ; iz<Nv; iz+=2*Nw+6) { if(isPMLs) PMLS##func<0><<<a,b,c,d>>>args; else func<0><<<a,b,c,d>>>args; } \
  for(int iz=Nw+3; iz<Nv; iz+=2*Nw+6) { if(isPMLs) PMLS##func<1><<<a,b,c,d>>>args; else func<1><<<a,b,c,d>>>args; } }
#endif
//#define IFPMLS(func,a,b,c,d,args) { if(!isPMLs) func<<<a,b,c,d>>>args; }
//#define IFPMLS(func,a,b,c,d,args) func<<<a,b,c,d>>>args;
template<int even> inline void Window::Dtorre(int ix, int Nt, int t0, double disbal[NDev], bool isPMLs, bool isTFSF) {
  if(Nt<=t0 || Nt<=0) return;
  DEBUG_PRINT(("Dtorre%d isPMLs=%d isTFSF=%d ix=%d, t0=%d Nt=%d wleft=%d\n", even, isPMLs, isTFSF, ix,t0,Nt, parsHost.wleft));
  const int Nw=Nzw-10;/*min(Nv-10,Nzw-10);*//*Nv/2*/; const int Nth=Nw+10;
  CHECK_ERROR( cudaSetDevice(0) );
  #ifdef TIMERS_ON
  cuTimer ttDm[NDev], ttDo[NDev];
  cudaStream_t stPMLbot; CHECK_ERROR( cudaStreamCreate(&stPMLbot) ); cudaStream_t stI; CHECK_ERROR( cudaStreamCreate(&stI   ) ); cuTimer ttPMLtop, ttI;
  cudaStream_t stDm[NDev],stDo[NDev]; for(int i=0;i<NDev;i++) { if(i!=0) CHECK_ERROR( cudaSetDevice(i) ); CHECK_ERROR( cudaStreamCreate(&stDm[i]) ); CHECK_ERROR( cudaStreamCreate(&stDo[i]) ); ttDm[i].created=0; ttDo[i].created=0; }
  cudaStream_t stPMLtop; CHECK_ERROR( cudaStreamCreate(&stPMLtop) ); cudaStream_t stX; CHECK_ERROR( cudaStreamCreate(&stX   ) ); cuTimer ttPMLbot, ttX;
  cudaStream_t stP; cuTimer ttP,ttPmpi; if(even==0) { cudaSetDevice(NDev-1); CHECK_ERROR( cudaStreamCreate(&stP   ) ); } else
                                        if(even==1) { cudaSetDevice(0     ); CHECK_ERROR( cudaStreamCreate(&stP   ) ); }
  cuTimer ttMPIa;
  #else//TIMER_S_ON not def
  cudaStream_t stPMLbot; CHECK_ERROR( cudaStreamCreate(&stPMLbot) ); cudaStream_t stI; CHECK_ERROR( cudaStreamCreate(&stI   ) );
  cudaStream_t stDm[NDev],stDo[NDev]; for(int i=0;i<NDev;i++) { if(i!=0) CHECK_ERROR( cudaSetDevice(i) ); CHECK_ERROR( cudaStreamCreate(&stDm[i]) ); CHECK_ERROR( cudaStreamCreate(&stDo[i]) ); }
  cudaStream_t stPMLtop; CHECK_ERROR( cudaStreamCreate(&stPMLtop) ); cudaStream_t stX; CHECK_ERROR( cudaStreamCreate(&stX   ) );
  cudaStream_t stP   ; if(even==0) { cudaSetDevice(NDev-1); CHECK_ERROR( cudaStreamCreate(&stP   ) ); } else
                       if(even==1) { cudaSetDevice(0     ); CHECK_ERROR( cudaStreamCreate(&stP   ) ); }
  #endif//TIMERS_ON
  CHECK_ERROR( cudaSetDevice(0) );

  int iym=0, iyp=0; 
  int Nblk=0;   iyp++;
  int Iy=iym, Xy, D1oy[NDev], D0oy[NDev], Dmy[NDev], DmBlk[NDev], Syb,Syt, SybBlk,SytBlk;
  int is_oneL[NDev], is_oneU[NDev], is_many[NDev], is_I[NDev], is_X[NDev], is_Sb[NDev], is_St[NDev], is_P[NDev];
  for(int i=0; i<NDev; i++) { is_oneL[i]=0; is_oneU[i]=0; is_many[i]=0; is_I[i]=0; is_X[i]=0; is_Sb[i]=0; is_St[i]=0; is_P[i]=0; }
  is_I[0]=1;
  iym=iyp; Nblk=0; while(iyp<Npmly/2) { iyp++; Nblk++; } if(Nblk>0) is_Sb[0]=1; Syb=iym; SybBlk=Nblk; 
  for(int idev=0,nextY=0; idev<NDev; idev++) {
    nextY+=NStripe[idev]; if(idev==NDev-1) nextY-=max(1,Npmly/2);
    if(idev!=0) {
    // Dtorre1 only
      if(iyp<nextY && even==1) is_oneL[idev]=1;
      D1oy[idev]=iyp; if(iyp<nextY) iyp++;
    }
    iym=iyp; Nblk=0;  while(iyp<nextY-(idev==NDev-1?0:1)) { iyp++; Nblk++; }
    // Main Region
    if(Nblk>0) is_many[idev]=1;
    Dmy[idev]=iym, DmBlk[idev]=Nblk;
    if(idev!=NDev-1) {
    // Dtorre0 only
      if(iyp<nextY && even==0) is_oneU[idev]=1;
      D0oy[idev]=iyp; if(iyp<nextY) iyp++;
    }
  }
  iym=iyp; Nblk=0;  while(iyp<Na-1) { iyp++; Nblk++; }
  if(Nblk>0) is_St[NDev-1]=1;
  is_X[NDev-1]=1;
  Syt=iym; SytBlk=Nblk; Xy=iyp;
  if(subnode!=0) {
    is_I [0]=0; if(even==1) is_P[0]=1;
    is_Sb[0]=0; DmBlk[0]+=SybBlk; Dmy[0]=Syb; 
  }
  if(subnode!=NasyncNodes-1) {
    is_X [NDev-1]=0; if(even==0) is_P[NDev-1]=1; 
    is_St[NDev-1]=0; DmBlk[NDev-1]+=SytBlk;
  }

  int mpirank = node*NasyncNodes+subnode;
  for(int idev=0; idev<NDev; idev++) {
    if(idev!=0) CHECK_ERROR( cudaSetDevice(idev) );
    if(is_oneL[idev] && even==1 &&  isTFSF ) IFPMLS(torreTFSF1 ,1          ,Nth,0,stDo[idev],ttDo[idev],(ix,D1oy[idev],iz,iz+Nw,Nt,t0))
    if(is_oneL[idev] && even==1 && !isTFSF ) IFPMLS(torreD1    ,1          ,Nth,0,stDo[idev],ttDo[idev],(ix,D1oy[idev],iz,iz+Nw,Nt,t0))
    if(is_oneL[idev] && even==1            ) bufsave<1><<<(Nv+Nw-1)/Nw,Nw,0,stDo[idev]>>>(ix,D1oy[idev],Nt,t0);
    if(is_oneU[idev] && even==0 &&  isTFSF ) IFPMLS(torreTFSF0 ,1          ,Nth,0,stDo[idev],ttDo[idev],(ix,D0oy[idev],iz,iz+Nw,Nt,t0))
    if(is_oneU[idev] && even==0 && !isTFSF ) IFPMLS(torreD0    ,1          ,Nth,0,stDo[idev],ttDo[idev],(ix,D0oy[idev],iz,iz+Nw,Nt,t0))
    if(is_oneU[idev] && even==0            ) bufsave<0><<<(Nv+Nw-1)/Nw,Nw,0,stDo[idev]>>>(ix,D0oy[idev],Nt,t0);
    if(is_I[idev]    && even==0 && Npmly==0) IFPMLS(torreId0   ,1          ,Nth,0,stI       ,ttI       ,(ix,Iy        ,iz,iz+Nw,Nt,t0))
    if(is_I[idev]    && even==0 && Npmly!=0) IFPMLS(torreIs0   ,1          ,Nth,0,stI       ,ttI       ,(ix,Iy        ,iz,iz+Nw,Nt,t0))
    if(is_I[idev]    && even==1 && Npmly==0) IFPMLS(torreId1   ,1          ,Nth,0,stI       ,ttI       ,(ix,Iy        ,iz,iz+Nw,Nt,t0))
    if(is_I[idev]    && even==1 && Npmly!=0) IFPMLS(torreIs1   ,1          ,Nth,0,stI       ,ttI       ,(ix,Iy        ,iz,iz+Nw,Nt,t0))
    if(is_X[idev]    && even==0 && Npmly==0) IFPMLS(torreXd0   ,1          ,Nth,0,stX       ,ttX       ,(ix,Xy        ,iz,iz+Nw,Nt,t0))
    if(is_X[idev]    && even==0 && Npmly!=0) IFPMLS(torreXs0   ,1          ,Nth,0,stX       ,ttX       ,(ix,Xy        ,iz,iz+Nw,Nt,t0))
    if(is_X[idev]    && even==1 && Npmly==0) IFPMLS(torreXd1   ,1          ,Nth,0,stX       ,ttX       ,(ix,Xy        ,iz,iz+Nw,Nt,t0))
    if(is_X[idev]    && even==1 && Npmly!=0) IFPMLS(torreXs1   ,1          ,Nth,0,stX       ,ttX       ,(ix,Xy        ,iz,iz+Nw,Nt,t0))
    if(is_P[idev]    && even==0            ) IFPMLS(torreD0    ,1          ,Nth,0,stP       ,ttP       ,(ix,Xy        ,iz,iz+Nw,Nt,t0))
    if(is_P[idev]    && even==1            ) IFPMLS(torreD1    ,1          ,Nth,0,stP       ,ttP       ,(ix,Iy        ,iz,iz+Nw,Nt,t0))
    if(is_P[idev]    && even==0            ) bufsave<0><<<(Nv+Nw-1)/Nw,Nw,0,stP       >>>(ix,Xy        ,Nt,t0);
    if(is_P[idev]    && even==1            ) bufsave<1><<<(Nv+Nw-1)/Nw,Nw,0,stP       >>>(ix,Iy        ,Nt,t0);
    if(is_Sb[idev]   && even==0            ) IFPMLS(torreS0    ,SybBlk     ,Nth,0,stPMLbot  ,ttPMLbot  ,(ix,Syb       ,iz,iz+Nw,Nt,t0))
    if(is_Sb[idev]   && even==1            ) IFPMLS(torreS1    ,SybBlk     ,Nth,0,stPMLbot  ,ttPMLbot  ,(ix,Syb       ,iz,iz+Nw,Nt,t0))
    if(is_St[idev]   && even==0            ) IFPMLS(torreS0    ,SytBlk     ,Nth,0,stPMLtop  ,ttPMLtop  ,(ix,Syt       ,iz,iz+Nw,Nt,t0))
    if(is_St[idev]   && even==1            ) IFPMLS(torreS1    ,SytBlk     ,Nth,0,stPMLtop  ,ttPMLtop  ,(ix,Syt       ,iz,iz+Nw,Nt,t0))
    if(is_many[idev] && even==0 && isTFSF  ) IFPMLS(torreTFSF0 ,DmBlk[idev],Nth,0,stDm[idev],ttDm[idev],(ix,Dmy[idev] ,iz,iz+Nw,Nt,t0))
    if(is_many[idev] && even==1 && isTFSF  ) IFPMLS(torreTFSF1 ,DmBlk[idev],Nth,0,stDm[idev],ttDm[idev],(ix,Dmy[idev] ,iz,iz+Nw,Nt,t0))
    if(is_many[idev] && even==0 && !isTFSF ) IFPMLS(torreD0    ,DmBlk[idev],Nth,0,stDm[idev],ttDm[idev],(ix,Dmy[idev] ,iz,iz+Nw,Nt,t0))
    if(is_many[idev] && even==1 && !isTFSF ) IFPMLS(torreD1    ,DmBlk[idev],Nth,0,stDm[idev],ttDm[idev],(ix,Dmy[idev] ,iz,iz+Nw,Nt,t0))
    if(is_oneL[idev] && even==1            ) DiamondRag::copyMbuf(idev, t0,Nt, stDo[idev]);
    if(is_oneU[idev] && even==0            ) DiamondRag::copyPbuf(idev, t0,Nt, stDo[idev]);
    #ifdef TIMERS_ON
    if(is_oneL[idev] && even==1 || is_oneU[idev] && even==0) ttDo[idev].record();
    #endif
  }
    if(NasyncNodes>1 && even==1            ) DiamondRag::prepTransM(mpirank, t0,Nt, stP);
    if(NasyncNodes>1 && even==0            ) DiamondRag::prepTransP(mpirank, t0,Nt, stP);
  #ifdef TIMERS_ON
  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    if(is_P[idev]) ttP.record();
  } CHECK_ERROR(cudaSetDevice(0));
  #endif

  CHECK_ERROR( cudaSetDevice(0) );

  float copytime=0;
  bool doSynccopy=0;
  if(!doneMemcopy) {
    doSynccopy=1;
    #ifdef TIMERS_ON
    for(int idev=0; idev<NDev; idev++) {
      CHECK_ERROR( cudaSetDevice(idev) ); 
      CHECK_ERROR( cudaEventRecord(copyEventStart[idev], streamCopy[idev]) );
    } CHECK_ERROR( cudaSetDevice(0) ); 
    if(even==0) MemcopyDtH(ix4copy);
    if(even==1) MemcopyHtD(ix4copy);
    for(int idev=0; idev<NDev; idev++) {
      CHECK_ERROR( cudaSetDevice(idev) ); 
      CHECK_ERROR( cudaEventRecord(copyEventEnd[idev], streamCopy[idev]) );
    } CHECK_ERROR( cudaSetDevice(0) );
    #else
    if(even==0) MemcopyDtH(ix4copy);
    if(even==1) MemcopyHtD(ix4copy);
    #endif
    if(even==1) doneMemcopy=true;
  }
  CHECK_ERROR( cudaStreamSynchronize(stP   ) );
  #ifdef TIMERS_ON
  timerP     += ttP.gettime_rec();
  #endif
  if(NasyncNodes>1) ampi_exch.exch(even, ix, t0, Nt, mpirank);
  if(NasyncNodes>1) ampi_exch.exch_sync();
  #ifdef TIMERS_ON
  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    if(is_P[idev]) ttPmpi.init(stP);
  } CHECK_ERROR(cudaSetDevice(0));
  #endif
  if(NasyncNodes>1 && even==1 ) DiamondRag::postTransM(mpirank, t0,Nt, stP);
  if(NasyncNodes>1 && even==0 ) DiamondRag::postTransP(mpirank, t0,Nt, stP);
  if(NasyncNodes>1) CHECK_ERROR( cudaStreamSynchronize(stP) );
  #ifdef TIMERS_ON
  for(int idev=0; idev<NDev; idev++) {
    CHECK_ERROR(cudaSetDevice(idev));
    if(is_P[idev]) ttPmpi.record();
  } CHECK_ERROR(cudaSetDevice(0));
  #endif

  if(doSynccopy) for(int idev=0; idev<NDev; idev++) CHECK_ERROR( cudaStreamSynchronize(streamCopy[idev]) );
  #ifdef TIMERS_ON
  if(doSynccopy) for(int idev=0; idev<NDev; idev++) {
    float copytime_idev;
    CHECK_ERROR( cudaEventElapsedTime(&copytime_idev, copyEventStart[idev], copyEventEnd[idev]) );
    copytime=max(copytime,copytime_idev);
  }
  timerCopy+= copytime;
  #endif

  //if(even==1) parsHost.drop.save(stPMLm);
  CHECK_ERROR( cudaStreamSynchronize(stPMLbot) ); 
  CHECK_ERROR( cudaStreamSynchronize(stPMLtop) );
  CHECK_ERROR( cudaStreamSynchronize(stI   ) );
  CHECK_ERROR( cudaStreamSynchronize(stX   ) );
  //CHECK_ERROR( cudaStreamSynchronize(stB   ) );
  for(int i=0;i<NDev;i++) CHECK_ERROR( cudaStreamSynchronize(stDo[i]) );
  int firsti=parsHost.iStep%NDev; double tt=omp_get_wtime(); CHECK_ERROR( cudaStreamSynchronize(stDm[firsti]) ); disbal[0]+=omp_get_wtime()-tt;
  for(int j=1;j<NDev;j++) { int i=(j+parsHost.iStep)%NDev; double tt=omp_get_wtime(); CHECK_ERROR( cudaStreamSynchronize(stDm[i]) ); disbal[j]+=omp_get_wtime()-tt; }

  CHECK_ERROR( cudaStreamDestroy(stPMLbot) );
  CHECK_ERROR( cudaStreamDestroy(stPMLtop) );
  CHECK_ERROR( cudaStreamDestroy(stI   ) ); 
  CHECK_ERROR( cudaStreamDestroy(stX   ) ); 
  //CHECK_ERROR( cudaStreamDestroy(stB   ) ); 
  CHECK_ERROR( cudaStreamDestroy(stP   ) ); 
  for(int i=0;i<NDev;i++) CHECK_ERROR( cudaStreamDestroy(stDo[i]) );
  for(int i=0;i<NDev;i++) CHECK_ERROR( cudaStreamDestroy(stDm[i]) );

  #ifdef TIMERS_ON
  timerPMLtop+= ttPMLtop.gettime_rec(); timerI+= ttI.gettime_rec(); for(int i=0;i<NDev;i++) timerDm[i]+= ttDm[i].gettime_rec();
  timerPMLbot+= ttPMLbot.gettime_rec(); timerX+= ttX.gettime_rec(); for(int i=0;i<NDev;i++) timerDo[i]+= ttDo[i].gettime_rec();
  timerP     += ttPmpi.gettime_rec();
  timerP     += ampi_exch.exch_time*1e3;
  
  float calctime = max(ttPMLtop.diftime,max(ttPMLbot.diftime,max(ttI.diftime,max(ttX.diftime,ttP.diftime+ttPmpi.diftime+ampi_exch.exch_time*1e3))));
  for(int i=0;i<NDev;i++) calctime=max(calctime,max(ttDm[i].diftime,ttDo[i].diftime));
  timerExec+= max(copytime, calctime);
  #endif
}
inline void Window::Dtorres(int ix, int Nt, int t0, double disbal[NDev], bool isPMLs, bool isTFSF) {
  Dtorre<0>(ix,Nt,t0,disbal,isPMLs,isTFSF); //cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
  Dtorre<1>(ix,Nt,t0,disbal,isPMLs,isTFSF); //cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
}

#ifdef MPI_ON
MPI_Request reqSp, reqSm, reqRp, reqRm, reqSp_pml, reqSm_pml, reqRp_pml, reqRm_pml;
MPI_Request reqSM_p2pbuf[NDev],reqSP_p2pbuf[NDev],reqRM_p2pbuf[NDev],reqRP_p2pbuf[NDev];
MPI_Status status;
int flagSp,flagRp,flagSm,flagRm,flagSp_pml,flagRp_pml,flagSm_pml,flagRm_pml;
mpi_message Window::mes[8];
int doWaitM,doWaitP;
//#define BLOCK_SEND
//#define MPI_NUDGE
//#define USE_MPI_THREADING

#ifndef USE_MPI_THREADING
#define WaitMPI(nreq,req,st) MPI_Wait(req,st)
//#define SendMPI(p,sz,tp,rnk,tag,world,req,nreq) MPI_Isend(p,sz,tp,rnk,tag,world,req);
#define SendMPI(p,sz,tp,rnk,tag,world,req,nreq) MPI_Send(p,sz,tp,rnk,tag,world);
#define RecvMPI(p,sz,tp,rnk,tag,world,req,nreq) MPI_Irecv(p,sz,tp,rnk,tag,world,req);
#else
#define WaitMPI(nreq,req,st) { mpi_message* mes = &window.mes[nreq]; \
       int s=pthread_join(mes->mpith,0); if(s!=0) printf("node %d: Error joining thread %ld retcode=%d\n",window.node,mes->mpith,s); }
static void* send_func(void* args){
  mpi_message *mes = (mpi_message*)args;
  MPI_Send(mes->buf,mes->count,mes->datatype,mes->dest,mes->tag,mes->comm);
  return 0;
}
static void* recv_func(void* args){
  mpi_message *mes = (mpi_message*)args;
  MPI_Status stat;
  MPI_Recv(mes->buf,mes->count,mes->datatype,mes->dest,mes->tag,mes->comm,&stat);
  return 0;
}
#define SendMPI(p,sz,tp,rnk,tag,world,req,nreq) {mpi_message* mes = &window.mes[nreq]; mes->set(p,sz,tp,rnk,tag,world); \
      if(pthread_create(&mes->mpith,0,send_func,(void*)mes)!=0) {printf("Error: cannot create thread for MPI_send %d node=%d\n",nreq,window.node); MPI_Abort(MPI_COMM_WORLD, 1);};}
#define RecvMPI(p,sz,tp,rnk,tag,world,req,nreq) {mpi_message* mes = &window.mes[nreq]; mes->set(p,sz,tp,rnk,tag,world); \
      if(pthread_create(&mes->mpith,0,recv_func,(void*)mes)!=0) {printf("Error: cannot create thread for MPI_recv %d node=%d\n",nreq,window.node); MPI_Abort(MPI_COMM_WORLD, 1);};}
#endif//USE_MPI_THREADING
#endif// MPI_ON
int calcStep(){
//  CHECK_ERROR( cudaDeviceSetSharedMemConfig ( cudaSharedMemBankSizeEightByte ) );
  if(parsHost.iStep==0) printf("Starting...\n");
  cuTimer t0; t0.init();
  int torreNum=0; double dropTime=0;
  CHECK_ERROR(cudaDeviceSynchronize());
  #ifdef TEST_RATE
  for(int ix=Ns-Ntime; ix>0; ix--) {
//    printf("ix=%d\n",ix);
    const int block_spacing = TEST_RATE;
    torreD0<<<(Na-2)/block_spacing,Nv>>>(ix, 1, Ntime, 0); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
    torreD1<<<(Na-2)/block_spacing,Nv>>>(ix, 1, Ntime, 0); cudaDeviceSynchronize(); CHECK_ERROR( cudaGetLastError() );
    torreNum++;
  }
  #else
  Window window; window.prepare();
  int node_shift=0; for(int inode=0; inode<window.node; inode++) node_shift+= mapNodeSize[inode]; node_shift-= Ns*window.node;
  int nsize=mapNodeSize[window.node]; int nL=node_shift; int nR=nL+nsize;
  #ifdef MPI_ON
  if(parsHost.iStep==0) {
    int wleftP=nR-Ns;
    int wleftM=nL;
    doWaitP=0; doWaitM=0;
    if(window.node!=window.Nprocs-1) {
      #ifndef MPI_TEST
      DEBUG_MPI(("timestamp %10.2f: Recv P (node %d) wleft=%d / tag %d, req %p\n", omp_get_wtime(), window.node, wleftP, 2, &reqRp));
      for(int idev=0; idev<NDev; idev++) {
        RecvMPI( parsHost.p2pBufM_host_rcv[idev], Ntime   , MPI_HLFRAGTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+0+window.Nprocs*10+2*idev+0, MPI_COMM_WORLD, &reqRM_p2pbuf[idev],);
        RecvMPI( parsHost.p2pBufP_host_rcv[idev], Ntime   , MPI_HLFRAGTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+0+window.Nprocs*10+2*idev+1, MPI_COMM_WORLD, &reqRP_p2pbuf[idev],);
      }
      RecvMPI(&window.data    [wleftP*Na   ], Ns*Na       , MPI_DMDRAGTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+0, MPI_COMM_WORLD, &reqRp    , 2);flagRp    =0;
      RecvMPI(&window.dataPMLa[wleftP*Npmly], Ns*Npmly    , MPI_RAGPMLTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+1, MPI_COMM_WORLD, &reqRp_pml, 6);flagRp_pml=0;
      doWaitP=1;
      #endif
    }
  }
  #endif//MPI_ON
  while(window.w0+Ns>=0) {
    #ifdef MPI_ON
    if( true ) {
      #ifdef DROP_DATA
      if(parsHost.wleft==nR-Ns-Ns-1) { cuTimer tdrop; tdrop.init(); parsHost.drop.drop( nsize-Ns            ,nsize   ,window.data,parsHost.iStep); dropTime+= tdrop.gettime(); }
      if(parsHost.wleft==nL-Ns-1   ) { cuTimer tdrop; tdrop.init(); parsHost.drop.drop((window.node==0)?0:Ns,nsize-Ns,window.data,parsHost.iStep); dropTime+= tdrop.gettime(); }
      #endif
      bool doSend[2] = {1,1}; bool doRecv[2] = {1,1};
      #ifdef MPI_TEST
      if(parsHost.iStep  -window.node<=0) { doSend[0]=0; doSend[1]=0; doRecv[1]=0; }
      if(parsHost.iStep+1-window.node<=0) { doRecv[0]=0; }
      #endif
      if(doWaitP && parsHost.wleft==nR+(Ns-Ntime-1)   ) {
        if(window.node!=window.Nprocs-1) DEBUG_MPI(("timestamp %10.2f: waiting P (node %d) wleft=%d / requests %p %p\n", omp_get_wtime(), window.node, parsHost.wleft, &reqRp,&reqSp)); 
        if(window.node!=window.Nprocs-1) { WaitMPI(2,&reqRp, &status);WaitMPI(6,&reqRp_pml, &status); flagRp=1;flagRp_pml=1;}
        if(window.node!=window.Nprocs-1) for(int idev=0;idev<NDev;idev++) {WaitMPI(,&reqRM_p2pbuf[idev], &status);WaitMPI(,&reqRP_p2pbuf[idev], &status);}
        if(window.node!=window.Nprocs-1) for(int idev=0;idev<NDev;idev++) {
          CHECK_ERROR(cudaMemcpy(parsHost.p2pBufM[idev],parsHost.p2pBufM_host_rcv[idev],Ntime*sizeof(halfRag),cudaMemcpyHostToDevice));
          CHECK_ERROR(cudaMemcpy(parsHost.p2pBufP[idev],parsHost.p2pBufP_host_rcv[idev],Ntime*sizeof(halfRag),cudaMemcpyHostToDevice));
        }
      }
      if(parsHost.wleft==nR-Ns-Ns-1 && window.node!=window.Nprocs-1) {
        if(doSend[1]) {
          DEBUG_MPI(("timestamp %10.2f: Send&Recv P(%d) (node %d) wleft=%d (tags %d,%d, reqs %p,%p)\n", omp_get_wtime(), parsHost.wleft+Ns, window.node, parsHost.wleft, 2+(parsHost.iStep+1)*2+0, 2+(parsHost.iStep+1)*2+1, &reqSp, &reqRp));
          SendMPI(&window.data    [(nR-Ns)*Na   ], doSend[1]*Ns*Na   , MPI_DMDRAGTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+0, MPI_COMM_WORLD, &reqSp    ,0);flagSp    =0;
          SendMPI(&window.dataPMLa[(nR-Ns)*Npmly], doSend[1]*Ns*Npmly, MPI_RAGPMLTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+1, MPI_COMM_WORLD, &reqSp_pml,4);flagSp_pml=0;
          DEBUG_MPI(("timestamp %10.2f: ok Send P(%d) (node %d) wleft=%d (tags %d,%d, reqs %p,%p)\n", omp_get_wtime(), parsHost.wleft+Ns, window.node, parsHost.wleft, 2+(parsHost.iStep+1)*2+0, 2+(parsHost.iStep+1)*2+0, &reqSp, &reqRp));
        }
        if(doRecv[1]) {
          for(int idev=0; idev<NDev; idev++) {
            RecvMPI( parsHost.p2pBufM_host_rcv[idev], doRecv[1]*Ntime, MPI_HLFRAGTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+window.Nprocs*10+2*idev+0, MPI_COMM_WORLD, &reqRM_p2pbuf[idev],);
            RecvMPI( parsHost.p2pBufP_host_rcv[idev], doRecv[1]*Ntime, MPI_HLFRAGTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+window.Nprocs*10+2*idev+1, MPI_COMM_WORLD, &reqRP_p2pbuf[idev],);
          }
          RecvMPI(&window.data    [(nR-Ns)*Na   ], doRecv[1]*Ns*Na   , MPI_DMDRAGTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+0, MPI_COMM_WORLD, &reqRp    ,2);flagRp    =0;
          RecvMPI(&window.dataPMLa[(nR-Ns)*Npmly], doRecv[1]*Ns*Npmly, MPI_RAGPMLTYPE, (window.node+1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+1, MPI_COMM_WORLD, &reqRp_pml,6);flagRp_pml=0;
          doWaitP=1;
        }
      }
//      if(doWaitM && parsHost.wleft==nL+Ns+(Ns-Ntime-1)   && parsHost.iStep!=0) { 
      if(doWaitM && parsHost.wleft==nR-1-Ns-((window.node==window.Nprocs-1)?(Ntime/2+1):Ns) && parsHost.iStep!=0) { 
        if(window.node!=0) DEBUG_MPI(("timestamp %10.2f: waiting M (node %d) wleft=%d / requests %p %p\n", omp_get_wtime(), window.node, parsHost.wleft, &reqRm, &reqSm)); 
        if(window.node!=0) { WaitMPI(3,&reqRm, &status);WaitMPI(7,&reqRm_pml, &status); flagRm=1;flagRm_pml=1;}
      }
      #ifdef MPI_NUDGE
      if((parsHost.wleft+Ns)%1==0) {
        if(doWaitP) if(                     window.node!=window.Nprocs-1) { if(!flagRp_pml || !flagRp) DEBUG_MPI(("timestamp %10.2f: testing recvP(%p) (node %d) wleft=%d\n", omp_get_wtime(), &reqRp, window.node, parsHost.wleft)); if(!flagRp) MPI_Test(&reqRp, &flagRp, &status); if(!flagRp_pml)MPI_Test(&reqRp_pml, &flagRp_pml, &status); }
        if(doWaitM) if(parsHost.iStep!=0 && window.node!=0              ) { if(!flagRm_pml || !flagRm) DEBUG_MPI(("timestamp %10.2f: testing recvM(%p) (node %d) wleft=%d\n", omp_get_wtime(), &reqRm, window.node, parsHost.wleft)); if(!flagRm) MPI_Test(&reqRm, &flagRm, &status); if(!flagRm_pml)MPI_Test(&reqRm_pml, &flagRm_pml, &status); }
      }
      #endif
      #ifdef MPI_TEST
      if(parsHost.iStep-window.node>0) {
      #endif
        ampi_exch.do_run=1;
        if(NasyncNodes>1) { if(sem_init(&ampi_exch.sem_calc, 0,0)==-1) printf("Error semaphore init errno=%d\n", errno);
                            if(sem_init(&ampi_exch.sem_mpi , 0,0)==-1) printf("Error semaphore init errno=%d\n", errno); }
        #pragma omp parallel num_threads(2)
        {
        if(omp_get_thread_num()==1) {
          window.calcDtorres(nL,nR, parsHost.wleft<nL && window.node!=0, parsHost.wleft>=nR-Ns && window.node!=window.Nprocs-1);
          ampi_exch.do_run=0; if(NasyncNodes>1) if(sem_post(&ampi_exch.sem_mpi)<0) printf("sem_post_mpi end error %d\n",errno); 
        }
          #pragma omp master
          if(NasyncNodes>1) { while(ampi_exch.do_run) ampi_exch.run(); if(sem_post(&ampi_exch.sem_calc)<0) printf("sem_post_calc end error %d\n",errno); }
        }
        if(NasyncNodes>1) { if(sem_destroy(&ampi_exch.sem_mpi )<0) printf("sem_destroy error %d\n",errno);
                            if(sem_destroy(&ampi_exch.sem_calc)<0) printf("sem_destroy error %d\n",errno); }
      #ifdef MPI_TEST
      }
      #endif

      if(parsHost.wleft==nL-Ns-1  && window.node!=0              ) {
        if(doSend[0]) {
          DEBUG_MPI(("timestamp %10.2f: Send&Recv M(%d) (node %d) wleft=%d (tags %d,%d, reqs %p,%p)\n", omp_get_wtime(), parsHost.wleft+Ns+1, window.node, parsHost.wleft, 2+(parsHost.iStep  )*2+0, 2+(parsHost.iStep+1)*2+0, &reqSm, &reqRm));
          for(int idev=0; idev<NDev; idev++) {
            CHECK_ERROR(cudaMemcpy(parsHost.p2pBufM_host_snd[idev],parsHost.p2pBufM[idev],Ntime*sizeof(halfRag),cudaMemcpyDeviceToHost));
            CHECK_ERROR(cudaMemcpy(parsHost.p2pBufP_host_snd[idev],parsHost.p2pBufP[idev],Ntime*sizeof(halfRag),cudaMemcpyDeviceToHost));
            SendMPI( parsHost.p2pBufM_host_snd[idev] , doSend[0]*Ntime   , MPI_HLFRAGTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep  )*2+window.Nprocs*10+2*idev+0, MPI_COMM_WORLD, &reqSM_p2pbuf[idev],);
            SendMPI( parsHost.p2pBufP_host_snd[idev] , doSend[0]*Ntime   , MPI_HLFRAGTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep  )*2+window.Nprocs*10+2*idev+1, MPI_COMM_WORLD, &reqSP_p2pbuf[idev],);
          }
          SendMPI(&window.data    [ nL    *Na   ], doSend[0]*Ns*Na   , MPI_DMDRAGTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep  )*2+0, MPI_COMM_WORLD, &reqSm    ,1);flagSm    =0;
          SendMPI(&window.dataPMLa[ nL    *Npmly], doSend[0]*Ns*Npmly, MPI_RAGPMLTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep  )*2+1, MPI_COMM_WORLD, &reqSm_pml,5);flagSm_pml=0;
          DEBUG_MPI(("timestamp %10.2f: ok Send M(%d) (node %d) wleft=%d (tags %d,%d, reqs %p,%p)\n", omp_get_wtime(), parsHost.wleft+Ns+1, window.node, parsHost.wleft, 2+(parsHost.iStep  )*2+0, 2+(parsHost.iStep+1)*2+0, &reqSm, &reqRm));
        }
        if(doRecv[0]) {
          RecvMPI(&window.data    [ nL    *Na   ], doRecv[0]*Ns*Na   , MPI_DMDRAGTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+0, MPI_COMM_WORLD, &reqRm,    3);flagRm    =0;
          RecvMPI(&window.dataPMLa[ nL    *Npmly], doRecv[0]*Ns*Npmly, MPI_RAGPMLTYPE, (window.node-1)*NasyncNodes+window.subnode, 2+(parsHost.iStep+1)*2+1, MPI_COMM_WORLD, &reqRm_pml,7);flagRm_pml=0;
          doWaitM=1;
        }
      }
    }
    #else//MPI_ON not def
    window.calcDtorres();
    #endif//MPI_ON
    window.synchronize();
  }
  window.finalize();
  #endif//TEST_RATE
  #if not defined MPI_ON && defined DROP_DATA
  cuTimer tdrop; tdrop.init(); parsHost.drop.drop(0,Np,parsHost.data,parsHost.iStep); dropTime+= tdrop.gettime();
  /*
  parsHost.drop.dump();
  #ifndef MPI_TEST
  if(0 && parsHost.iStep%(10*window.Nprocs)==0) parsHost.drop.sync();
  #endif
  */
  #endif

  double calcTime=t0.gettime();
  unsigned long int yee_cells = 0;
  double overhead=0;
  #ifndef TEST_RATE
  yee_cells = NDT*NDT*Ntime*(unsigned long long)(Nv*((Na+1-NDev)*NasyncNodes+1-NasyncNodes))*Np;
  overhead = window.RAMcopytime/window.GPUcalctime;
  printf("Step %d /node %d/ subnode %d/: Time %9.09f ms |drop %3.03f%% ||rate %9.09f GYee_cells/sec |total grid %dx%dx%d=%ld cells | isTFSF=%d\n",
  parsHost.iStep, window.node, window.subnode, calcTime, 100*dropTime/calcTime, 
  1.e-9*yee_cells/(calcTime*1.e-3), NDT*Np,NDT*((Na+1-NDev)*NasyncNodes+1-NasyncNodes),Nv,yee_cells/Ntime, (parsHost.iStep+1)*Ntime*dt<shotpoint.tStop );
//  for(int idev=0;idev<NDev;idev++) printf("%3.03f%% ", 100*window.disbal[idev]/window.GPUcalctime);
  #ifdef TIMERS_ON
  printf("         |waitings%d %5.05f",(parsHost.iStep)%NDev,1.e3*window.disbal[0]); for(int idev=1; idev<NDev; idev++) printf(", %5.05f", 1.e3*window.disbal[idev]); printf("\n");
  for(int idev=0; idev<NDev; idev++) printf("         |timers(Step,node,subnode,device): %d %d %d %d | PMLbot PMLtop I X Do Dmi P Copy Exec:| %.02f %.02f %.02f %.02f %.02f %.02f %.02f %.02f %.02f\n",
                                       parsHost.iStep,window.node,window.subnode,idev,
                                       window.timerPMLbot, window.timerPMLtop, window.timerI, window.timerX, window.timerDo[idev], window.timerDm[idev], window.timerP, window.timerCopy, window.timerExec);
  #endif//TIMERS_ON
  #else//if def TEST_RATE 
  yee_cells = NDT*NDT*Ntime*(unsigned long long)(Nv*((Na-2)/TEST_RATE))*torreNum;
  printf("Step %d: Time %9.09f ms |drop %3.03f%% |rate %9.09f %d %d %d %d (GYee cells/sec,Np,Na,Nv,Ntime) |isTFSF=%d \n", parsHost.iStep, calcTime, 100*dropTime/calcTime, 1.e-9*yee_cells/(calcTime*1.e-3), Np,Na,Nv,Ntime, (parsHost.iStep+1)*Ntime*dt<shotpoint.tStop );
  #endif//TEST_RATE
  #ifdef MPI_ON
  double AllCalcTime;
  MPI_Reduce(&calcTime, &AllCalcTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if(window.node==0 && 0) printf("===(%3d)===AllCalcTime %9.09f sec |rate %9.09f GYee_cells/sec\n", parsHost.iStep, AllCalcTime*1e-3, 1.e-9*yee_cells/(AllCalcTime*1.e-3) );
  #endif
  fflush(stdout);
  parsHost.iStep++;
  copy2dev(parsHost, pars);
  return 0; 
}
