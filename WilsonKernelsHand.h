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
    #if defined (SVE) || defined(AVX512)
      #if defined(SVE)
        #pragma message ("RRII kernel using SVE ACLE")
        #include "arch/sve/rrii/SVE_rrii.h"
        //#include "arch/sve/rrii/rrii_kernel.cpp"
      #endif
      #if defined(AVX512)
        #pragma message ("RRII kernel using AVX512 intrinsics")
        #include "arch/avx512/rrii/AVX512_rrii.h"
      #endif
    #else
      #pragma message ("RRII kernel undefined")
      #pragma error
    #endif
  #else
    #if defined (SVE) || defined(AVX512)
      #if defined(SVE)
        #pragma message ("RRII kernel using GCC vectors and SVE ACLE for prefetching")
        #include "arch/sve/rrii/SVEGCCVectorsPF.h"
      #endif
      #if defined(AVX512)
        #pragma message ("RRII kernel using GCC vectors")
        #include "arch/avx512/rrii/AVX512GCCVectors.h"
      #endif
    #else
      #pragma message ("GridBench RRII kernel using GCC vectors")
      //#include "WilsonKernelsHandCpu.h"
      #pragma error
    #endif
  #endif

#else // RIRI

  #ifdef INTRIN
    #if defined (SVE) || defined(AVX512)
      #if defined(SVE)
        #pragma message ("RIRI kernel using SVE ACLE")
        #include "arch/sve/riri/SVE_riri.h"
        //#include "arch/sve/riri/riri_kernel.cpp"
      #endif
      #if defined(AVX512)
        #pragma message ("RIRI kernel using AVX512 intrinsics")
        #include "WilsonKernelsHandCpu.h"
      #endif
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
  return dslash_kernel_cpu_JL(nrep,Up,outp,inp,nbr,nsite,Ls,prm,psi_pf_dist_L1,psi_pf_dist_L2,u_pf_dist_L2);
}
#endif
