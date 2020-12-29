#pragma once

#include "Macros.h"

#define Xp (0)
#define Yp (1)
#define Zp (2)
#define Tp (3)
#define Xm (4)
#define Ym (5)
#define Zm (6)
#define Tm (7)

#ifdef VGPU
#include "WilsonKernelsHandGpu.h"

template<class Simd>
double dslash_kernel(int nrep,Simd *Up,Simd *outp,Simd *inp,uint64_t *nbr,uint64_t nsite,uint64_t Ls,uint8_t *prm)
{
  return dslash_kernel_gpu(nrep,Up,outp,inp,nbr,nsite,Ls,prm);
}
#else
#ifdef RRII

  #ifdef INTRIN
    #ifdef SVE
      #pragma message ("RRII kernel using SVE ACLE")
      #include "arch/sve/rrii/SVE_rrii.h"
      //#include "WilsonKernelsHandCpuSVETemplate.h"
      //#include "WilsonKernelsHandCpuSVETemplateDebug.h"
      //#include "WilsonKernelsHandCpuSVETemplateDebug1.h"
    #else
      #pragma message ("RRII kernel undefined")
      #pragma error
    #endif
  #else
    #ifdef SVE
      #pragma message ("RRII kernel using GCC vectors and SVE ACLE for prefetching")
      //#include "WilsonKernelsHandCpuSVETemplateDebug.h"
      #include "arch/sve/rrii/SVEGCCVectorsPF.h"
    #else
      #pragma message ("GridBench RRII kernel using GCC vectors")
      #include "WilsonKernelsHandCpu.h"
    #endif
  #endif

#else

  #ifdef INTRIN
    #ifdef SVE
      #pragma message ("RIRI kernel using SVE ACLE")
      #include "arch/sve/riri/SVE_riri.h"
    #else
      #pragma message ("RIRI kernel undefined")
      #pragma error
    #endif

  #else

    #pragma message ("GridBench RIRI kernel")
    #include "WilsonKernelsHandCpu.h"

  #endif

#endif

template<class Simd>
double dslash_kernel(int nrep,Simd *Up,Simd *outp,Simd *inp,uint64_t *nbr,uint64_t nsite,uint64_t Ls,uint8_t *prm,int psi_pf_dist_L1, int psi_pf_dist_L2, int u_pf_dist_L2)
{
  return dslash_kernel_cpu(nrep,Up,outp,inp,nbr,nsite,Ls,prm,psi_pf_dist_L1,psi_pf_dist_L2,u_pf_dist_L2);
}
#endif
