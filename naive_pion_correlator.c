/***********************************************************************
 *
 * Copyright (C) 2012 Carsten Urbach, Albert Deuzeman, Bartosz Kostrzewa
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 *
 * naive pion correlator for twisted mass QCD
 *
 *******************************************************************************/

#define MAIN_PROGRAM

#include"lime.h"
#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#ifdef MPI
#include <mpi.h>
#endif
#ifdef OMP
# include <omp.h>
#endif
#include "global.h"
#include "git_hash.h"
#include "getopt.h"
#include "linalg_eo.h"
#include "geometry_eo.h"
#include "start.h"
/*#include "eigenvalues.h"*/
#include "measure_gauge_action.h"
#ifdef MPI
#include "xchange/xchange.h"
#endif
#include <io/utils.h>
#include "read_input.h"
#include "mpi_init.h"
#include "sighandler.h"
#include "boundary.h"
#include "solver/solver.h"
#include "init/init.h"
#include <smearing/control.h>
#include "invert_eo.h"
#include "monomial/monomial.h"
#include "ranlxd.h"
#include "phmc.h"
#include "operator/D_psi.h"
#include "little_D.h"
#include "reweighting_factor.h"
#include "linalg/convert_eo_to_lexic.h"
#include "block.h"
#include "operator.h"
#include "sighandler.h"
#include "solver/dfl_projector.h"
#include "solver/generate_dfl_subspace.h"
#include "prepare_source.h"
#include <io/params.h>
#include <io/gauge.h>
#include <io/spinor.h>
#include <io/utils.h>
#include "solver/dirac_operator_eigenvectors.h"
#include "P_M_eta.h"
#include "operator/tm_operators.h"
#include "operator/Dov_psi.h"
#include "gettime.h"
#include "dirty_shameful_business.h"

extern int nstore;
int check_geometry();

static void usage();
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename);
static void set_default_filenames(char ** input_filename, char ** filename);

int main(int argc, char *argv[])
{
  FILE *parameterfile = NULL;
  int j, i, ix = 0, isample = 0, op_id = 0;
  char datafilename[206];
  char parameterfilename[206];
  char conf_filename[50];
  char * input_filename = NULL;
  char * filename = NULL;
  double plaquette_energy;
  double oneover2kappasqSV;
  spinor **s, *s_;

#ifdef _KOJAK_INST
#pragma pomp inst init
#pragma pomp inst begin(main)
#endif

#if (defined SSE || defined SSE2 || SSE3)
  signal(SIGILL, &catch_ill_inst);
#endif

  DUM_DERI = 8;
  DUM_MATRIX = DUM_DERI + 5;
#if ((defined BGL && defined XLC) || defined _USE_TSPLITPAR)
  NO_OF_SPINORFIELDS = DUM_MATRIX + 3;
#else
  NO_OF_SPINORFIELDS = DUM_MATRIX + 3;
#endif

  verbose = 0;
  g_use_clover_flag = 0;

#ifdef MPI

#  ifdef OMP
  int mpi_thread_provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);
#  else
  MPI_Init(&argc, &argv);
#  endif

  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
#else
  g_proc_id = 0;
#endif

  process_args(argc,argv,&input_filename,&filename);
  set_default_filenames(&input_filename, &filename);

  /* Read the input file */
  if( (j = read_input(input_filename)) != 0) {
    fprintf(stderr, "Could not find input file: %s\nAborting...\n", input_filename);
    exit(-1);
  }

#ifdef OMP
  init_openmp();
#endif

  /* this DBW2 stuff is not needed for the inversion ! */
  if (g_dflgcr_flag == 1) {
    even_odd_flag = 0;
  }
  g_rgi_C1 = 0;
  if (Nsave == 0) {
    Nsave = 1;
  }

  if (g_running_phmc) {
    NO_OF_SPINORFIELDS = DUM_MATRIX + 8;
  }

  tmlqcd_mpi_init(argc, argv);

  g_dbw2rand = 0;

  /* starts the single and double precision random number */
  /* generator                                            */
  start_ranlux(rlxd_level, random_seed);
  
  /* Allocate needed memory */
  initialize_gauge_buffers(5);
  initialize_adjoint_buffers(5);
  init_smearing();

  /* initialize set of 24 spinors to hold the result of the 12 inversions and their conjugates */

  spinor* M_inv[4][3];
  spinor* M_trans_inv[4][3];

  spinor** S;
  spinor* S_memory;

  allocate_spinor_field_array(&S, &S_memory, VOLUME, 24);

  for(unsigned int spin=0; spin < 4; ++spin) {
    for(unsigned int col=0; col < 3; ++col) {
      M_inv[spin][col] = S[3*spin+col];
      M_trans_inv[spin][col] = S[3*spin+col+12];
    }
  }

  double* Cpp = calloc(T,sizeof(double));

  /* we need to make sure that we don't have even_odd_flag = 1 */
  /* if any of the operators doesn't use it                    */
  /* in this way even/odd can still be used by other operators */
  for(j = 0; j < no_operators; j++) if(!operator_list[j].even_odd_flag) even_odd_flag = 0;

#ifndef MPI
  g_dbw2rand = 0;
#endif

#ifdef _GAUGE_COPY
  j = init_gauge_field(VOLUMEPLUSRAND, 1);
#else
  j = init_gauge_field(VOLUMEPLUSRAND, 0);
#endif
  if (j != 0) {
    fprintf(stderr, "Not enough memory for gauge_fields! Aborting...\n");
    exit(-1);
  }
  j = init_geometry_indices(VOLUMEPLUSRAND);
  if (j != 0) {
    fprintf(stderr, "Not enough memory for geometry indices! Aborting...\n");
    exit(-1);
  }
  if (no_monomials > 0) {
    if (even_odd_flag) {
      j = init_monomials(VOLUMEPLUSRAND / 2, even_odd_flag);
    }
    else {
      j = init_monomials(VOLUMEPLUSRAND, even_odd_flag);
    }
    if (j != 0) {
      fprintf(stderr, "Not enough memory for monomial pseudo fermion fields! Aborting...\n");
      exit(-1);
    }
  }
  if (even_odd_flag) {
    j = init_spinor_field(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS);
  }
  else {
    j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS);
  }
  if (j != 0) {
    fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
    exit(-1);
  }

  if (g_running_phmc) {
    j = init_chi_spinor_field(VOLUMEPLUSRAND / 2, 20);
    if (j != 0) {
      fprintf(stderr, "Not enough memory for PHMC Chi fields! Aborting...\n");
      exit(-1);
    }
  }

  g_mu = g_mu1;

  if (g_cart_id == 0) {
    /*construct the filenames for the observables and the parameters*/
    strncpy(datafilename, filename, 200);
    strcat(datafilename, ".data");
    strncpy(parameterfilename, filename, 200);
    strcat(parameterfilename, ".para");

    parameterfile = fopen(parameterfilename, "w");
    write_first_messages(parameterfile, "invert", git_hash);
    fclose(parameterfile);
  }

  /* define the geometry */
  geometry();

  /* define the boundary conditions for the fermion fields */
  boundary(g_kappa);

  phmc_invmaxev = 1.;

  init_operators();

  /* this could be maybe moved to init_operators */
#ifdef _USE_HALFSPINOR
  j = init_dirac_halfspinor();
  if (j != 0) {
    fprintf(stderr, "Not enough memory for halffield! Aborting...\n");
    exit(-1);
  }
  if (g_sloppy_precision_flag == 1) {
    j = init_dirac_halfspinor32();
    if (j != 0)
    {
      fprintf(stderr, "Not enough memory for 32-bit halffield! Aborting...\n");
      exit(-1);
    }
  }
#  if (defined _PERSISTENT)
  if (even_odd_flag)
    init_xchange_halffield();
#  endif
#endif

  for (j = 0; j < Nmeas; j++) {
    sprintf(conf_filename, "%s.%.4d", gauge_input_filename, nstore);
    if (g_cart_id == 0) {
      printf("#\n# Trying to read gauge field from file %s in %s precision.\n",
            conf_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
      fflush(stdout);
    }
    if( (i = read_gauge_field(conf_filename)) !=0) {
      fprintf(stderr, "Error %d while reading gauge field from %s\n Aborting...\n", i, conf_filename);
      exit(-2);
    }
  
    if (g_cart_id == 0) {
      printf("# Finished reading gauge field.\n");
      fflush(stdout);
    }

#ifdef MPI
    xchange_gauge(g_gauge_field);
#endif

    /*compute the energy of the gauge field*/
    plaquette_energy = measure_gauge_action(_AS_GAUGE_FIELD_T(g_gauge_field));

    if (g_cart_id == 0) {
      printf("# The computed plaquette value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
      fflush(stdout);
    }

    if (g_cart_id == 0) {
      fprintf(stdout, "#\n"); /*Indicate starting of the operator part*/
    }

    for (int stype = 0; stype < no_smearings_operator; ++stype)
    {
      smear(smearing_control_operator[stype], g_gf);
      double new_plaquette = measure_gauge_action(smearing_control_operator[stype]->result);

      if (g_cart_id == 0)
      {
        printf("# After smearing type %d, the plaquette value is %e.\n", stype, new_plaquette / (6.*VOLUME*g_nproc));
        fflush(stdout);
      }

      ohnohack_remap_g_gauge_field(smearing_control_operator[stype]->result);

      for(op_id = 0; op_id < no_operators; op_id++) {
        if (operator_list[op_id].smearing != stype)
        {
          continue; /* if this operator is not smeared with the current stype, skip */
        }

        operator * optr = &operator_list[op_id]; 
        boundary(optr->kappa);
        g_kappa = optr->kappa; 
        g_mu = optr->mu;
        oneover2kappasqSV = 1.0/(2*optr->kappa*optr->kappa*g_nproc_x*g_nproc_y*g_nproc_z*LX*LY*LZ);

        for (ix = 0; ix < 12; ix++) {
          if (g_cart_id == 0) {
            fprintf(stdout, "#\n"); /*Indicate starting of new index*/
          }
          prepare_source(nstore, isample, ix, op_id, read_source_flag, source_location);

          optr->iterations = invert_eo( optr->prop0, optr->prop1, optr->sr0, optr->sr1,
                                optr->eps_sq, optr->maxiter,optr->solver, optr->rel_prec,
                                0, optr->even_odd_flag,optr->no_extra_masses, optr->extra_masses, optr->id);
    
          /* check result */
          M_full(g_spinor_field[DUM_DERI], g_spinor_field[DUM_DERI+1], optr->prop0, optr->prop1);
          diff(g_spinor_field[DUM_DERI], g_spinor_field[DUM_DERI], optr->sr0, VOLUME / 2);
          diff(g_spinor_field[DUM_DERI+1], g_spinor_field[DUM_DERI+1], optr->sr1, VOLUME / 2);

          double nrm1 = square_norm(g_spinor_field[DUM_DERI], VOLUME / 2, 1);
          double nrm2 = square_norm(g_spinor_field[DUM_DERI+1], VOLUME / 2, 1);
          optr->reached_prec = nrm1 + nrm2;

          printf("# Reached precision for spin %d color %d: %e\n",ix/3,ix%3,optr->reached_prec);

          convert_eo_to_lexic(M_inv[ix/3][ix%3],optr->prop0,optr->prop1);
        
        }

        /* do transpose in spin-colour space only (conjugate is taken in spinor product below) */
        complex double *ptr;
        complex double *ptr2;
        double t_begin = gettime();
        #ifdef OMP
        #pragma omp parallel for private(ptr,ptr2)
        #endif
        for(int x=0;x<VOLUME;++x) {
          for(int spin1=0;spin1<4;++spin1) {
            for(int col1=0;col1<3;++col1) {
              for(int spin2=0;spin2<4;++spin2) {
                for(int col2=0;col2<3;++col2) {
                  ptr = (complex double*)&M_trans_inv[spin1][col1][x]+spin2*3+col2;
                  ptr2 = (complex double*)&M_inv[spin2][col2][x]+spin1*3+col1;
                  *ptr = *ptr2;
                }
              }
            }
          }
        }
        double t_spent = gettime() - t_begin;
        printf("## Hermitian conjugation took %e seconds.\n",t_spent);


        t_begin=gettime();
        #ifdef OMP
        #pragma omp parallel
        {
        #endif
  
        #ifdef OMP
        #pragma omp for
        #endif
        for(int t=0;t<T;++t){
          Cpp[t]=0;
          double kc=0,ks=0,tr=0,tt=0,ts=0;
          int j = g_ipt[t][0][0][0];
          for(int x=j;x<j+LX*LY*LZ;++x){
            for(int spin=0; spin<4; ++spin){
              for(int col=0; col<3; ++col){
                tr=_spinor_prod_re(M_inv[spin][col][j],M_trans_inv[spin][col][j])+kc;
                ts=tr+ks;
                tt=ts-ks;
                ks=ts;
                kc=tr-tt;
              }
            }
          }
          Cpp[t] = (kc+ks)*oneover2kappasqSV;
        }
    
        #ifdef OMP
        } /* OpenMP parallel closing brace */
        #endif
    
        t_spent = gettime() - t_begin;
        printf("## Correlator computation took: %e seconds\n",t_spent);
     
        /* store correlator to file */
        char f_correlator_filename[100];
        snprintf(f_correlator_filename,99,"Cpp.data.%02d.%06d",op_id,nstore);
        FILE* f_correlator = fopen(f_correlator_filename,"w");

        if(f_correlator != NULL) { 
          for(int t=0;t<T;++t){
            fprintf(f_correlator,"%d %e\n",t,Cpp[t]);
          }
        }
        fclose(f_correlator);
      }
      ohnohack_remap_g_gauge_field(g_gf);
    }
    nstore += Nsave;
  }

  free(Cpp);
  free_spinor_field_array(&S_memory);

  free(S);
  return_gauge_field(&g_gf);

#ifdef MPI
  MPI_Finalize();
#endif
#ifdef OMP
  free_omp_accumulators();
#endif

  free_blocks();
  free_dfl_subspace();
  free_geometry_indices();
  free_spinor_field();

  free_chi_spinor_field();
  finalize_gauge_buffers();
  finalize_adjoint_buffers();
  finalize_smearing();

  free(filename);
  free(input_filename);

  return(0);
  
  
#ifdef _KOJAK_INST
#pragma pomp inst end(main)
#endif
}

static void usage()
{
  fprintf(stdout, "Computation of the naive connected pion correlator in Wilson twisted mass QCD\n");
  fprintf(stdout, "Version %s \n\n", PACKAGE_VERSION);
  fprintf(stdout, "Please send bug reports to %s\n", PACKAGE_BUGREPORT);
  fprintf(stdout, "Usage:   invert [options]\n");
  fprintf(stdout, "Options: [-f input-filename]\n");
  fprintf(stdout, "         [-o output-filename]\n");
  fprintf(stdout, "         [-v] more verbosity\n");
  fprintf(stdout, "         [-h|-? this help]\n");
  fprintf(stdout, "         [-V] print version information and exit\n");
  exit(0);
}

static void process_args(int argc, char *argv[], char ** input_filename, char ** filename) {
  int c;
  while ((c = getopt(argc, argv, "h?vVf:o:")) != -1) {
    switch (c) {
      case 'f':
        *input_filename = calloc(200, sizeof(char));
        strncpy(*input_filename, optarg, 200);
        break;
      case 'o':
        *filename = calloc(200, sizeof(char));
        strncpy(*filename, optarg, 200);
        break;
      case 'v':
        verbose = 1;
        break;
      case 'V':
        if(g_proc_id == 0) {
          fprintf(stdout,"%s %s\n",PACKAGE_STRING,git_hash);
        }
        exit(0);
        break;
      case 'h':
      case '?':
      default:
        if( g_proc_id == 0 ) {
          usage();
        }
        break;
    }
  }
}

static void set_default_filenames(char ** input_filename, char ** filename) {
  if( *input_filename == NULL ) {
    *input_filename = calloc(28, sizeof(char));
    strcpy(*input_filename,"naive_pion_correlator.input");
  }
  
  if( *filename == NULL ) {
    *filename = calloc(7, sizeof(char));
    strcpy(*filename,"output");
  } 
}

