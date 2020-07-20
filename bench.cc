
// Invoke dslash.s - test for compiler-gsnerated code
#include <stdio.h>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <strings.h>
#include <math.h>
#include <chrono>
#include <cassert>

#include "Simd.h"
#include "WilsonKernelsHand.h"


#ifdef __x86_64__
#define __SSC_MARK(A) __asm__ __volatile__ ("movl %0, %%ebx; .byte 0x64, 0x67, 0x90 " ::"i"(A):"%ebx")
#else
#define __SSC_MARK(A)
#endif

///////////////////////////////////////
// Preinitialised arrays
///////////////////////////////////////

#ifdef VGPU
#include "arch/avx512/static_data.h" // 64 Byte layout
#endif

#ifdef GEN
#include "arch/sse/static_data.h"
#endif

#ifdef SSE4
#include "arch/sse/static_data.h"
#endif

#if defined(AVX1) || defined (AVXFMA) || defined(AVX2) || defined(AVXFMA4)
#include "arch/avx/static_data.h"
#endif

#ifdef AVX512
#include "arch/avx512/static_data.h"
#endif

#ifdef RRII
#include "arch/gen64/static_data.h"
#endif


#define  FMT std::dec
int main(int argc, char* argv[])
{

  ////////////////////////////////////////////////////////////////////
  // Option 2: copy from static arrays
  ////////////////////////////////////////////////////////////////////
  uint64_t nreplica = 16;
  uint64_t umax   = nsite*18*8 *vComplexD::Nsimd(); 
  uint64_t fmax   = nsite*24*Ls*vComplexD::Nsimd(); 
  uint64_t nbrmax = nsite*Ls*8;
  uint64_t vol    = nsite*Ls*vComplexD::Nsimd(); 
  
  printf("Nsimd %d\n",vComplexD::Nsimd());


  Vector<double>   U(umax*nreplica);       
  Vector<double>   Psi(fmax*nreplica);     
  Vector<double>   Phi(fmax*nreplica);     
  Vector<double>   Psi_cpp(fmax*nreplica); 
  Vector<uint64_t> nbr(nsite*Ls*8*nreplica); 
  Vector<uint8_t>  prm(nsite*Ls*8*nreplica); 

  for(int replica=0;replica<nreplica;replica++){
    int u=replica*umax;
    int f=replica*fmax;
    int n=replica*nbrmax;
    bcopy(U_static,&U[u],umax*sizeof(double));
    bzero(&Psi[f],fmax*sizeof(double));
    bcopy(Phi_static,&Phi[f],fmax*sizeof(double));
    bcopy(Psi_cpp_static,&Psi_cpp[f],fmax*sizeof(double));
    bcopy(nbr_static,&nbr[n],nbrmax*sizeof(uint64_t));
    bcopy(prm_static,&prm[n],nbrmax*sizeof(uint8_t));
    for(int nn=0;nn<nbrmax;nn++){
      nbr[nn+n]+=nsite*Ls; // Shift the neighbour indexes to point to this replica
    }
  }
  
  Vector<float>   fU(umax*nreplica);       
  Vector<float>   fPsi(fmax*nreplica);     
  Vector<float>   fPhi(fmax*nreplica);
  Vector<float>   fPsi_cpp(fmax*nreplica); 

#ifdef RRII
  const int Nsimd = vComplexD::Nsimd();
  for(uint32_t r=0;r<nreplica;r++){
  for(uint32_t ss=0;ss<nsite;ss++){
  for(uint32_t s=0;s<Ls;s++){
  for(uint32_t sc=0;sc<12;sc++){
    for(uint32_t n=0;n<vComplexD::Nsimd();n++){
      for(uint32_t ri=0;ri<2;ri++){
	int idx = ss*Ls*24*Nsimd
	        +     s*24*Nsimd
    	        +      sc*2*Nsimd;
	int ridx= idx+r*nsite*Ls*24*Nsimd;
	Phi     [ridx + ri*Nsimd + n ] =     Phi_static[idx + n*2 + ri];
	Psi_cpp [ridx + ri*Nsimd + n ] = Psi_cpp_static[idx + n*2 + ri];
	fPhi    [ridx + ri*Nsimd + n ] =     Phi_static[idx + n*2 + ri];
	fPsi_cpp[ridx + ri*Nsimd + n ] = Psi_cpp_static[idx + n*2 + ri];
      }
    }
  }}}}
  std::cout << "Remapped Spinor data\n" ;
  for(uint32_t r=0;r<nreplica;r++){
  for(uint32_t ss=0;ss<nsite*9*8;ss++){
    for(uint32_t n=0;n<vComplexD::Nsimd();n++){
      for(uint32_t ri=0;ri<2;ri++){
	U [r*Nsimd*2*nsite*9*8+ss*Nsimd*2 + ri*Nsimd + n ] = U_static[ss*Nsimd*2 + n*2 + ri];
	fU[r*Nsimd*2*nsite*9*8+ss*Nsimd*2 + ri*Nsimd + n ] = U_static[ss*Nsimd*2 + n*2 + ri];
      }
    }
  }}
  std::cout << "Remapped Gauge data\n";
#endif
  
  std::cout << std::endl;
  std::cout << "Calling dslash_kernel "<<std::endl;


  double flops = 1320.0*vol*nreplica;
  int nrep=100; // cache warm
#define DOUBLE
#ifdef DOUBLE
  double usec = dslash_kernel<vComplexD>(nrep,
			   (vComplexD *)&U[0],
			   (vComplexD *)&Psi[0],
			   (vComplexD *)&Phi[0],
			   &nbr[0],
			   nsite*nreplica,
			   Ls,
			   &prm[0]);
#else
  double usec = dslash_kernel<vComplexF>(nrep,
			   (vComplexF *)&fU[0],
			   (vComplexF *)&fPsi[0],
			   (vComplexF *)&fPhi[0],
			   &nbr[0],
			   nsite*nreplica,
			   Ls,
			   &prm[0]);

  // Copy back to double
  for(uint64_t i=0; i<fmax*nreplica;i++){
    Psi[i]=fPsi[i];
  }
#endif

  std::cout << std::endl;
#ifdef DOUBLE
  std::cout <<"\t"<< nrep*flops/usec/1000. << " Gflop/s in double precision; kernel call "<<usec/nrep <<" microseconds "<<std::endl;
#else
  std::cout <<"\t"<< nrep*flops/usec/1000. << " Gflop/s in single precision; kernel call "<<usec/nrep <<" microseconds "<<std::endl;
#endif
  std::cout << std::endl;

  // Check results

  vComplexD *Psi_p = (vComplexD *) &Psi[0];
  vComplexD *Psi_cpp_p = (vComplexD *) &Psi_cpp[0];
  for(uint64_t r=0; r<nreplica;r++){
    double err=0;
    double nref=0;
    double nres=0;
    for(uint64_t ii=0; ii<fmax;ii++){
      uint64_t i=ii+r*fmax;
      err += (Psi_cpp[i]-Psi[i])*(Psi_cpp[i]-Psi[i]);
      nres += Psi[i]*Psi[i];
      nref += Psi_cpp[i]*Psi_cpp[i];
    };
    std::cout<< "normdiff "<< err<< " ref "<<nref<<" result "<<nres<<std::endl;
    for(int ii=0;ii<20;ii++){
      uint64_t i=ii+r*fmax;
      std::cout<< i<<" ref "<<Psi_cpp[i]<< " result "<< Psi[i]<<std::endl;
    }
  }
  //  assert(err <= 1.0e-6);  
  return 0;
}
