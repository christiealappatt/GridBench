#include "../../../WilsonKernelsHand.h"

//typedef Simd<complex<double>  , SIMD_CDtype> vComplexD;

template<class vComplexD>
double dslash_kernel_cpu_JL(int nrep,vComplexD *Up,vComplexD *outp,vComplexD *inp,uint64_t *nbr,uint64_t nsite,uint64_t Ls,uint8_t *prm,int psi_pf_dist_L1, int psi_pf_dist_L2, int u_pf_dist_L2)
{
    return dslash_kernel_cpu<vComplexD>(nrep, Up, outp, inp, nbr, nsite, Ls, prm, psi_pf_dist_L1, psi_pf_dist_L2, u_pf_dist_L2);
}
