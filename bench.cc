#define  DOUBLE
#define DATA_SIMD 8  // Size in static data
#define EXPAND_SIMD 8  // Size in static data

// Invoke dslash.s - test for compiler-gsnerated code
#include <stdio.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <strings.h>
#include <iomanip>
#include <math.h>
#include <chrono>
#include <cassert>

#include "Simd.h"
#include "WilsonKernelsHand.h"

#ifdef OMP
#include <omp.h>
// some OMP function calls don't work on my laptop, use alternative
int omp_thread_count() {
    int n = 0;
    #pragma omp parallel reduction(+:n)
    n += 1;
    return n;
}
#endif

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

#ifdef RIRI
#include "arch/gen64/static_data.h"
#endif

// read CPU frequency from file
double read_freq() {

  double freq = 1.0;
  std::ifstream ifile("freq.txt", std::ios::in);

  if (!ifile.is_open()) {
        std::cout << "There was a problem opening the input file!\n";
        exit(1);
  }

  ifile >> freq;
  ifile.close();

  return freq;
}

#define  FMT std::dec
int main(int argc, char* argv[])
{
  double frequency = read_freq();

  ////////////////////////////////////////////////////////////////////
  // Option 2: copy from static arrays
  ////////////////////////////////////////////////////////////////////
  uint64_t nreplica = 1;
  uint64_t nbrmax = nsite*Ls*8;

  uint64_t umax   = nsite*18*8 *vComplexD::Nsimd();
  uint64_t fmax   = nsite*24*Ls*vComplexD::Nsimd();
  uint64_t vol    = nsite*Ls*vComplexD::Nsimd();

  std::cout << "Usage: bench.* [<replicas=1 of 8x8x8x8xLs lattice, Ls=8 is fixed>] [<iterations=1000>]" << std::endl << std::endl;

  nreplica = argc > 1 ? atoi(argv[1]) : 1;
  int nrep = argc > 2 ? atoi(argv[2]) : 1000;

  // check iterations
  assert(nrep > 0);
  // check if nreplica is > 0 and power of 2
  assert(nreplica > 0);
  assert( (nreplica & (nreplica - 1)) == 0 );

  int threads = 1;
//#ifdef _OPENMP
#ifdef OMP
threads = omp_thread_count();
#endif

  std::cout << "Frequency  = " << frequency << " GHz" << std::endl;
  std::cout << "Threads    = " << threads << std::endl;
  std::cout << "Nsimd      = " << vComplexD::Nsimd() << std::endl;
  std::cout << "Replicas   = " << nreplica << std::endl;
  std::cout << "Iterations = " << nrep << std::endl;

  std::cout << std::endl;

  std::cout << "  Ls       = " << Ls << std::endl;
  std::cout << "  nsite    = " << nsite * nreplica << std::endl;
  std::cout << "  volume   = " << vol * nreplica << std::endl;
  // decompose
  int Latt[5] = {1,1,1,1,Ls};
  int j = 0;
  auto v = nsite * vComplexD::Nsimd() * nreplica;
  while(v > 1) {
    v /= 2;
    Latt[j % 4] *= 2;
    j++;
  }
  std::cout << "           = " << Latt[3] << " x " << Latt[2] << " x " << Latt[1] << " x " << Latt[0] << " x " << Latt[4] << std::endl;

  std::cout << std::endl;

  uint64_t udata = umax * sizeof(double) * nreplica;
  uint64_t fdata = fmax * sizeof(double) * nreplica;
  std::cout << "  U        = " << udata / (1024. * 1024.) << " MiB" << std::endl;
  std::cout << "  psi in   = " << fdata / (1024. * 1024.) << " MiB" << std::endl;
  std::cout << "  psi out  = " << fdata / (1024. * 1024.) << " MiB" << std::endl;
  std::cout << "  total    = " << (udata + 2 * fdata) / (1024. * 1024.) << " MiB" << std::endl;

  std::cout << std::endl;

  std::cout << "Grid reference benchmark: " << std::endl;
  std::cout << "srun -n 1 ./Benchmark_dwf --mpi 1.1.1.1 --grid "
    << Latt[3] << "." << Latt[2] << "." << Latt[1] << "." << Latt[0]
    << " -Ls 8 --dslash-asm --threads " << threads << " | grep \": mflop/s =\" "
    << std::endl << std::endl;

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
    bcopy(nbr_static,&nbr[n],nbrmax*sizeof(uint64_t));
    bcopy(prm_static,&prm[n],nbrmax*sizeof(uint8_t));
    for(int nn=0;nn<nbrmax;nn++){
      nbr[nn+n]+=nsite*Ls*replica; // Shift the neighbour indexes to point to this replica
    }
  }

  Vector<float>   fU(umax*nreplica);
  Vector<float>   fPsi(fmax*nreplica);
  Vector<float>   fPhi(fmax*nreplica);
  Vector<float>   fPsi_cpp(fmax*nreplica);

  //std::cout << "&U   = " << &U[0] << std::endl;
  //std::cout << "&Psi = " << &Psi[0] << std::endl;
  //std::cout << "&Phi = " << &Phi[0] << std::endl;

  assert(vComplexD::Nsimd()==EXPAND_SIMD);
  //assert(vComplexF::Nsimd()==EXPAND_SIMD);
  const int Nsimd  = EXPAND_SIMD;
  const int NNsimd = DATA_SIMD;
  const int nsimd_replica=Nsimd/NNsimd;
  std::cout << " Expanding SIMD width by "<<nsimd_replica<<"x"<<std::endl;
#ifdef RRII
#define VEC_IDX(ri,n,nn) (ri*Nsimd+nn*NNsimd+n)
#else
#define VEC_IDX(ri,n,nn) (nn*NNsimd*2 + n*2 +ri)
#endif
  for(uint32_t r=0;r<nreplica;r++){
  for(uint32_t ss=0;ss<nsite;ss++){
  for(uint32_t s=0;s<Ls;s++){
  for(uint32_t sc=0;sc<12;sc++){
    for(uint32_t n=0;n<NNsimd;n++){
    for(uint32_t nn=0;nn<nsimd_replica;nn++){
      for(uint32_t ri=0;ri<2;ri++){
	int idx = ss*Ls*24*NNsimd
	        +     s*24*NNsimd
    	        +     sc*2*NNsimd;
	int ridx= idx*nsimd_replica+r*nsite*Ls*24*Nsimd;
	Phi     [ridx + VEC_IDX(ri,n,nn) ] =     Phi_static[idx + n*2 + ri];
	Psi_cpp [ridx + VEC_IDX(ri,n,nn) ] = Psi_cpp_static[idx + n*2 + ri];
	fPhi    [ridx + VEC_IDX(ri,n,nn) ] =     Phi_static[idx + n*2 + ri];
	fPsi_cpp[ridx + VEC_IDX(ri,n,nn) ] = Psi_cpp_static[idx + n*2 + ri];
      }
    }}
  }}}}
  std::cout << "Remapped Spinor data\n" ;
  for(uint32_t r=0;r<nreplica;r++){
  for(uint32_t ss=0;ss<nsite*9*8;ss++){
    for(uint32_t n=0;n<NNsimd;n++){
    for(uint32_t nn=0;nn<nsimd_replica;nn++){
      for(uint32_t ri=0;ri<2;ri++){
	U [r*Nsimd*2*nsite*9*8+ss*Nsimd*2 + VEC_IDX(ri,n,nn) ] = U_static[ss*NNsimd*2 + n*2 + ri];
	fU[r*Nsimd*2*nsite*9*8+ss*Nsimd*2 + VEC_IDX(ri,n,nn) ] = U_static[ss*NNsimd*2 + n*2 + ri];
      }
    }}
  }}
  std::cout << "Remapped Gauge data\n";

  std::cout << std::endl;
  std::cout << "Calling dslash_kernel "<<std::endl;


  double flops = 1320.0*vol*nreplica;
  //int nrep=1000; // cache warm
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

  // 8 * 12 + 8 * 9 in
  // 12 out
  double sec = usec / 1000000.;

  uint64_t total_data  = udata + 2 * fdata;
  double tp10          = ((total_data * nrep) / sec) / (1000. * 1000. * 1000.);
  double tp2           = ((total_data * nrep) / sec) / (1024. * 1024. * 1024.);
  // should parametrize % peak by vector size in hardware!
  double percent_peak  = 100. * ((nrep*flops/usec/1000.)/frequency/threads) / (2.*2.*8);
  double cycles        = sec * frequency * 1000. * 1000. * 1000.;
  double gflops_per_s  = nrep*flops/usec/1000.;
  double usec_per_Ls   = usec/nrep/(nsite* nreplica)/Ls;
  double cycles_per_Ls = cycles/nrep/(nsite* nreplica)/Ls;

  std::cout <<"XX\t"<< gflops_per_s << " GFlops/s DP; kernel per vector site "
    << usec_per_Ls <<" usec / " << cycles_per_Ls << " cycles" <<std::endl;

  std::cout <<"YY\t"<< gflops_per_s/frequency << " Flops/cycle DP; kernel per vector site "
    << usec_per_Ls <<" usec / " << cycles_per_Ls << " cycles" <<std::endl;

  std::cout <<"ZZ\t"<< gflops_per_s/frequency/threads << " Flops/cycle DP per thread; kernel per vector site "
    << usec_per_Ls * threads <<" usec / " << cycles_per_Ls * threads << " cycles" <<std::endl;

  std::cout <<std::endl;

  std::cout <<"XX\t"<< gflops_per_s/frequency/threads/ EXPAND_SIMD << " Flops/cycle DP per thread; kernel per single site "
    << usec_per_Ls * threads/ EXPAND_SIMD << " usec / "
    << cycles_per_Ls * threads / EXPAND_SIMD << " cycles" <<std::endl;

  std::cout << std::endl;

  std::cout <<"\t"<< percent_peak << " % peak" << std::endl;

  std::cout << std::endl;

  total_data = (8 * 9 + 8 * 12 + 12) * 2 * sizeof(double) * vComplexD::Nsimd() * nsite * nreplica * Ls;
  tp10 = ((total_data * nrep) / sec) / (1000. * 1000. * 1000.);
  tp2  = ((total_data * nrep) / sec) / (1024. * 1024. * 1024.);
  std::cout <<"\t"<< tp10 << "  GB/s RF throughput (base 10)" <<std::endl;
  std::cout <<"\t"<< tp2  << " GiB/s RF throughput (base  2)" <<std::endl;
  //std::cout << "\tdata transfer RF per iteration = " << total_data / (1024. * 1024) << " MiB" << std::endl;

  std::cout <<"\t"<< nrep*flops/usec/1000. << " Gflop/s in double precision; kernel call "<<usec/nrep <<" microseconds "<<std::endl;
#else
  std::cout <<"\t"<< nrep*flops/usec/1000. << " Gflop/s in single precision; kernel call "<<usec/nrep <<" microseconds "<<std::endl;
#endif
  std::cout << std::endl;

  // One liner result output
  std::cout << "# Threads     Replicas    Volume    GFlop/s    % peak    Cycles per single site    Cycles per vector site" << std::endl;
  std::cout
    << threads       << "  "
    << nreplica      << "  "
    << Latt[3] << "x" << Latt[2] << "x" << Latt[1] << "x" << Latt[0] << "x" << Latt[4] << "  "
    << gflops_per_s  << "  "
    << percent_peak  << "  "
    << cycles_per_Ls * threads / EXPAND_SIMD << "  "
    << cycles_per_Ls * threads << "  "
    << "XX1" << std::endl << std::endl;

  // Check results

  vComplexD *Psi_p = (vComplexD *) &Psi[0];
  vComplexD *Psi_cpp_p = (vComplexD *) &Psi_cpp[0];
  double err=0;
  for(uint64_t r=0; r<nreplica;r++){
    double nref=0;
    double nres=0;
    for(uint64_t ii=0; ii<fmax;ii++){
      uint64_t i=ii+r*fmax;
      err += (Psi_cpp[i]-Psi[i])*(Psi_cpp[i]-Psi[i]);
      nres += Psi[i]*Psi[i];
      nref += Psi_cpp[i]*Psi_cpp[i];
    };
    if (r == 0) {
      std::cout<< "normdiff "<< err<< " ref "<<nref<<" result "<<nres<<std::endl;
    }
    for(int ii=0;ii<64;ii++){
      uint64_t i=ii+r*fmax;
      //std::cout<< i<<" ref "<<Psi_cpp[i]<< " result "<< Psi[i]<<std::endl;
    }
  }
  assert(err <= 1.0e-6);
  return 0;
}
