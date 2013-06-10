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
#include <io/spinor.h>
#include <io/gauge.h>
#include <io/utils.h>
#include "solver/dirac_operator_eigenvectors.h"
#include "P_M_eta.h"
#include "operator/tm_operators.h"
#include "operator/Dov_psi.h"
#include "gettime.h"
#include "dirty_shameful_business.h"
#include "measurements.h"

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
  double t_begin, t_spent;

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

  g_rgi_C1 = 0;
  if (Nsave == 0) {
    Nsave = 1;
  }

  tmlqcd_mpi_init(argc, argv);
  
  /* Allocate needed memory */
  initialize_gauge_buffers(12);
  initialize_adjoint_buffers(6);

  /* starts the single and double precision random number */
  /* generator                                            */
  start_ranlux(rlxd_level, random_seed);

#ifdef _GAUGE_COPY
  j = init_gauge_field(VOLUMEPLUSRAND, 1);
#else
  j = init_gauge_field(VOLUMEPLUSRAND, 0);
#endif
  
  j = init_geometry_indices(VOLUMEPLUSRAND);
  if (j != 0) {
    fprintf(stderr, "Not enough memory for geometry indices! Aborting...\n");
    exit(-1);
  }
  /* init some e/o-spinor fields for read_spinor */
  j = init_spinor_field(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS);
  if (j != 0) {
    fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
    exit(-1);
  }

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

  init_operators();

  operator *optr = &operator_list[0];
  int no_extra_masses = optr->no_extra_masses;
  
  /* allocate memory to hold the spinors that will be read from file and the KK correlators */
  
  spinor** S;
  spinor* S_memory;

  allocate_spinor_field_array(&S, &S_memory, VOLUME, no_extra_masses+1);

  double *sCkk, *Ckk;

#ifdef MPI
  sCkk = (double*) calloc(T, sizeof(double));
  if(g_mpi_time_rank == 0) {
    Ckk = (double*) calloc(g_nproc_t*T, sizeof(double));
  }
#else
  Ckk = (double*) calloc(T, sizeof(double));
#endif

  char spinor_filename[200];

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

  for (int meas = 0; meas < Nmeas; ++meas) {
    /* read the corresponding gauge field */
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
    
    for(int mass = 0; mass <= no_extra_masses; ++mass ) {
      /* set the correct mass for the operator */
      g_mu = (mass==0) ? optr->mu : optr->extra_masses[mass];
      sprintf(spinor_filename, "source.00.%04d.00000.cgmms.%02d.%1d", nstore, mass, SourceInfo.type);
      if (g_cart_id == 0) {
        printf("#\n# Trying to read propagator for mass %d from file %s.\n",mass,spinor_filename);
        fflush(stdout);
      }
      if( (i = read_spinor(g_spinor_field[0],g_spinor_field[1],spinor_filename,0)) !=0) {
        fprintf(stderr, "Error %d while reading propagator from %s\n Aborting...\n", i, spinor_filename);
        exit(-2);
      }
      if (g_cart_id == 0) {
        printf("# Finished reading spinor for mass %d.\n",mass);
        fflush(stdout);
      }
      printf("kappa: %lf\n",g_kappa);
      /* CGMMS stores (Q^+ Q^-)^(-1), multiply with Q^- to get correct spinor */
      Qtm_minus_psi(g_spinor_field[0],g_spinor_field[1]);
      convert_eo_to_lexic(S[mass],g_spinor_field[0],g_spinor_field[1]);
    }
   
     /* correalator computation */

    double res, mres;
    for(int mass = 0; mass <= no_extra_masses; ++mass ) {
      t_begin = gettime();
    
      /* now we sum only over local space for every t */
      for(int t = 0; t < T; t++) {
        int j = g_ipt[t][0][0][0];
        res = 0.;
        for(int i = j; i < j+LX*LY*LZ; i++) {
          res += _spinor_prod_re(S[mass][i], S[0][i]);
        }

#if defined MPI
      MPI_Reduce(&res, &mres, 1, MPI_DOUBLE, MPI_SUM, 0, g_mpi_time_slices);
      res = mres;
      sCkk[t] = +res/(g_nproc_x*LX)/(g_nproc_y*LY)/(g_nproc_z*LZ)/2./optr->kappa/optr->kappa;
#else
      Ckk[t] = +res/(g_nproc_x*LX)/(g_nproc_y*LY)/(g_nproc_z*LZ)/2./optr->kappa/optr->kappa;
#endif
      }

#ifdef MPI
      /* some gymnastics needed in case of parallelisation */
      if(g_mpi_time_rank == 0) {
        MPI_Gather(sCkk, T, MPI_DOUBLE, Ckk, T, MPI_DOUBLE, 0, g_mpi_SV_slices);
      }
#endif

    t_spent = gettime() - t_begin;
    printf("## Correlator computation took: %e seconds\n",t_spent);

    char f_correlator_filename[100];
    snprintf(f_correlator_filename,99,"Ckk.data.%02d.%06d",mass,nstore);
    FILE* f_correlator = fopen(f_correlator_filename,"w");

    if(f_correlator != NULL) { 
      for(int t=0;t<T;++t){
        fprintf(f_correlator,"%d %e\n",t,Ckk[t]);
      }
    }
    fclose(f_correlator);
    
    } /* loop over masses */
    nstore += Nsave;
  } /* loop over measurements */

  return_gauge_field(&g_gf);

  free(Ckk);
#ifdef MPI
  free(sCkk);
#endif

  free_spinor_field_array(&S_memory);
  free(S);

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
  finalize_smearing();
  finalize_gauge_buffers();
  finalize_adjoint_buffers();

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
    *input_filename = calloc(32, sizeof(char));
    strcpy(*input_filename,"timeslice_kaon_correlator.input");
  }
  
  if( *filename == NULL ) {
    *filename = calloc(7, sizeof(char));
    strcpy(*filename,"output");
  } 
}

