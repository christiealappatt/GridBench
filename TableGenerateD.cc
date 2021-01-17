#include <stdio.h>
#include <vector>
#include <iostream>
#include <stdint.h>
#include <strings.h>
#include <math.h>

#include <Grid/Grid.h>
using namespace Grid;

//const int Ls=8;

#define  FMT std::dec
int main(int argc, char* argv[])
{
  ////////////////////////////////////////////////////////////////////
  // Option 1: use Grid to build reference data and tables
  ////////////////////////////////////////////////////////////////////
  Grid_init(&argc,&argv);

  Coordinate latt4 = GridDefaultLatt();
  int Ls=8;
  int tofile = 1;
  for(int i=0;i<argc;i++) {
    if(std::string(argv[i]) == "-Ls"){
      std::stringstream ss(argv[i+1]); ss >> Ls;
    }
    if(std::string(argv[i]) == "-static"){
      tofile = 0;
    }
  }

  GridLogLayout();


  GridCartesian         * UGrid   = SpaceTimeGrid::makeFourDimGrid(GridDefaultLatt(),
								   GridDefaultSimd(Nd,vComplex::Nsimd()),
								   GridDefaultMpi());
  GridRedBlackCartesian * UrbGrid = SpaceTimeGrid::makeFourDimRedBlackGrid(UGrid);
  GridCartesian         * FGrid   = SpaceTimeGrid::makeFiveDimGrid(Ls,UGrid);
  GridRedBlackCartesian * FrbGrid = SpaceTimeGrid::makeFiveDimRedBlackGrid(Ls,UGrid);

  std::vector<int> seeds4({1,2,3,4});
  std::vector<int> seeds5({5,6,7,8});

  GridParallelRNG          RNG4(UGrid);  RNG4.SeedFixedIntegers(seeds4);
  GridParallelRNG          RNG5(FGrid);  RNG5.SeedFixedIntegers(seeds5);

  LatticeFermion src   (FGrid); random(RNG5,src);
  LatticeFermion result(FGrid); result=Zero();
  LatticeFermion result_cpp(FGrid); result_cpp=Zero();
  LatticeFermion    ref(FGrid);    ref=Zero();
  LatticeFermion    err(FGrid);

  uint64_t nsite = UGrid->oSites();

  LatticeGaugeField Umu(UGrid);   SU3::HotConfiguration(RNG4,Umu);
  LatticeDoubledGaugeField Uds(UGrid);

  RealD M5  = 1.0;
  RealD mass= 0.1;

  DomainWallFermionR Dw(Umu,*FGrid,*FrbGrid,*UGrid,*UrbGrid,mass,M5);

  Dw.Dhop(src,ref,0);

  Uds = Dw.Umu;
  uint64_t umax   = nsite*18*8*vComplex::Nsimd();
  uint64_t fmax   = nsite*24*Ls*vComplex::Nsimd();
  uint64_t nbrmax = nsite*Ls*8;
  //auto Uds_v = Uds.View();
  autoView(  Uds_v, Uds, CpuRead );
  //auto result_v = result.View();
  autoView(  result_v, result, CpuRead );
  //auto src_v = src.View();
  autoView(  src_v, src, CpuRead );
  //auto ref_v = ref.View();
  autoView(  ref_v, ref, CpuRead );

  Real * U   = (Real *)& Uds_v[0];
  Real * Psi = (Real *)& result_v[0];
  Real * Phi = (Real *)& src_v[0];
  Real * Psi_cpp = (Real *)& ref_v[0];


  std::cout << " umax " <<umax<<std::endl;
  std::cout << " fmax " <<fmax<<std::endl;

  Vector<uint64_t> lo (nsite,0);
  Vector<uint64_t> nbr(nbrmax,0);
  Vector<uint8_t>     prm(nbrmax,0);
  for(int site=0;site<nsite;site++){
    for(int s=0;s<Ls;s++){
      int idx=s+Ls*site;
      for(int mu=0;mu<8;mu++){
	int jdx=mu+s*8+8*Ls*site;

	int ptype;
	StencilEntry *SE= Dw.Stencil.GetEntry(ptype,mu,idx);
	uint64_t offset = SE->_offset;
	int local       = SE->_is_local; assert(local==1);
	int perm        = SE->_permute;

	nbr[jdx] = offset;
	prm[jdx] = perm;
      }
    }
  }

  {
    /////////////////////////////
    // write static data to disk
    /////////////////////////////
    if (!tofile) {
      FILE *fp = fopen("static_data.cc","w");

      fprintf(fp,"#include <stdint.h>\n");
      fprintf(fp,"double U_static[] = { \n ");
      for(uint64_t n=0;n<umax;n++) fprintf(fp,"    %16.8le, \n",U[n]);
      fprintf(fp,"    0}; \n ");

      fprintf(fp,"double Phi_static[] = { \n ");
      for(uint64_t n=0;n<fmax;n++) fprintf(fp,"    %16.8le, \n",Phi[n]);
      fprintf(fp,"    0}; \n ");

      fprintf(fp,"double Psi_cpp_static[] = { \n ");
      for(uint64_t n=0;n<fmax;n++) fprintf(fp,"    %16.8le, \n",Psi_cpp[n]);
      fprintf(fp,"    0}; \n ");

      fprintf(fp,"uint64_t nbr_static[] = { \n ");
      for(uint64_t n=0;n<nbrmax;n++) fprintf(fp,"    0x%llx, \n",nbr[n]);
      fprintf(fp,"    0}; \n ");

      fprintf(fp,"uint8_t prm_static[] = { \n ");
      for(uint64_t n=0;n<nbrmax;n++) fprintf(fp,"    0x%x, \n",(unsigned)prm[n]);
      fprintf(fp,"    0}; \n ");

      fclose(fp);
    } else {
      // ---- binary data ----
      std::cout << "writing data_riri.bin" << std::endl;
      FILE *fp = std::fopen("data_riri.bin", "w");
      std::fwrite(&nsite, sizeof(uint64_t), 1, fp);
      int l4[4] = {latt4[0],latt4[1],latt4[2],latt4[3]};
      std::fwrite(&l4[0], sizeof(int), 4, fp);
      std::fwrite(&Ls, sizeof(int), 1, fp);

      std::fwrite(&U[0], sizeof(double), umax, fp);
      std::fwrite(&Phi[0], sizeof(double), fmax, fp);
      std::fwrite(&Psi_cpp[0], sizeof(double), fmax, fp);
      std::fwrite(&nbr[0], sizeof(uint64_t), nbrmax, fp);
      std::fwrite(&prm[0], sizeof(uint8_t), nbrmax, fp);
      fclose(fp);
   }
  }

   {
    /////////////////////////////
    // write static data to disk
    /////////////////////////////
    if (!tofile) { // static data in static_data.cc
      FILE *fp = fopen("static_data.h","w");
      fprintf(fp,"// header file for use of static data, static_data.cc\n");
      fprintf(fp,"#include <stdint.h>\n");
      fprintf(fp,"const uint64_t nsite = %llu ; \n",nsite);
      fprintf(fp,"const int latt4 = {%d,%d,%d,%d} ; \n",latt4[0],latt4[1],latt4[2],latt4[3]);
      fprintf(fp,"const int Ls = %d ; \n",Ls);

      fprintf(fp,"extern double   U_static[] ; \n");
      fprintf(fp,"extern double   Phi_static[] ; \n");
      fprintf(fp,"extern double   Psi_cpp_static[] ; \n");
      fprintf(fp,"extern uint64_t nbr_static[] ; \n");
      fprintf(fp,"extern uint8_t  prm_static[] ; \n");
      fclose(fp);
    }
  }
  return 0;
}
