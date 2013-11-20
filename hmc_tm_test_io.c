/***********************************************************************
 *
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
 *                                             2013 Bartosz Kostrzewa
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
 *
 * I/O Testing code based on the HMC 
 *
 *******************************************************************************/
#define MAIN_PROGRAM
#include "lime.h"
#if HAVE_CONFIG_H
#include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
#ifdef MPI
# include <mpi.h>
#endif
#ifdef OMP
# include <omp.h>
#endif
#include "global.h"
#include "git_hash.h"
#include <io/params.h>
#include <io/gauge.h>
#include "getopt.h"
#include "ranlxd.h"
#include "geometry_eo.h"
#include "start.h"
#include "measure_gauge_action.h"
#include "measure_rectangles.h"
#ifdef MPI
# include "xchange/xchange.h"
#endif
#include "read_input.h"
#include "mpi_init.h"
#include "sighandler.h"
#include "update_tm.h"
#include "init/init.h"
#include "test/check_geometry.h"
#include "boundary.h"
#include "phmc.h"
#include "solver/solver.h"
#include "monomial/monomial.h"
#include "integrator.h"
#include "sighandler.h"
#include "measurements.h"

extern int nstore;

int const rlxdsize = 105;

static void usage();
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename);
static void set_default_filenames(char ** input_filename, char ** filename);

int main(int argc,char *argv[]) {

  FILE *parameterfile=NULL, *countfile=NULL;
  char *filename = NULL;
  char datafilename[206];
  char parameterfilename[206];
  char gauge_filename[50];
  char nstore_filename[50];
  char tmp_filename[50];
  char *input_filename = NULL;
  int status = 0, accept = 0;
  int j,ix,mu, trajectory_counter=0;
  unsigned int const io_max_attempts = 5; /* Make this configurable? */
  unsigned int const io_timeout = 5; /* Make this configurable? */
  struct timeval t1;

  /* Energy corresponding to the Gauge part */
  double plaquette_energy = 0., rectangle_energy = 0.;
  /* Acceptance rate */
  int Rate=0;
  /* Do we want to perform reversibility checks */
  /* See also return_check_flag in read_input.h */
  int return_check = 0;
  /* For getopt */
  int c;

  paramsXlfInfo *xlfInfo;

/* For online measurements */
  measurement * meas;
  int imeas;
  
#ifdef _KOJAK_INST
#pragma pomp inst init
#pragma pomp inst begin(main)
#endif

#if (defined SSE || defined SSE2 || SSE3)
  signal(SIGILL,&catch_ill_inst);
#endif

  strcpy(gauge_filename,"conf.save");
  strcpy(nstore_filename,".nstore_counter");
  strcpy(tmp_filename, ".conf.tmp");

  verbose = 1;
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
  set_default_filenames(&input_filename,&filename);

  /* Read the input file */
  if( (status = read_input(input_filename)) != 0) {
    fprintf(stderr, "Could not find input file: %s\nAborting...\n", input_filename);
    exit(-1);
  }

#ifdef OMP
  init_openmp();
#endif

  DUM_DERI = 4;
  DUM_SOLVER = DUM_DERI+1;
  DUM_MATRIX = DUM_SOLVER+6;
  if(g_running_phmc) {
    NO_OF_SPINORFIELDS = DUM_MATRIX+8;
  }
  else {
    NO_OF_SPINORFIELDS = DUM_MATRIX+6;
  }
  DUM_BI_DERI = 6;
  DUM_BI_SOLVER = DUM_BI_DERI+7;

  DUM_BI_MATRIX = DUM_BI_SOLVER+6;
  NO_OF_BISPINORFIELDS = DUM_BI_MATRIX+6;

  tmlqcd_mpi_init(argc, argv);

  if(nstore == -1) {
    countfile = fopen(nstore_filename, "r");
    if(countfile != NULL) {
      j = fscanf(countfile, "%d %d %s\n", &nstore, &trajectory_counter, gauge_input_filename);
      if(j < 1) nstore = 0;
      if(j < 2) trajectory_counter = 0;
      fclose(countfile);
    }
    else {
      nstore = 0;
      trajectory_counter = 0;
    }
  }
  
#ifndef MPI
  g_dbw2rand = 0;
#endif

  g_mu = g_mu1;
  
#ifdef _GAUGE_COPY
  status = init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 1);
#else
  status = init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 0);
#endif
  /* need temporary gauge field for gauge reread checks and in update_tm */
  status += init_gauge_tmp(VOLUME);

  if (status != 0) {
    fprintf(stderr, "Not enough memory for gauge_fields! Aborting...\n");
    exit(0);
  }
  j = init_geometry_indices(VOLUMEPLUSRAND + g_dbw2rand);
  if (j != 0) {
    fprintf(stderr, "Not enough memory for geometry_indices! Aborting...\n");
    exit(0);
  }
  if(even_odd_flag) {
    j = init_spinor_field(VOLUMEPLUSRAND/2, NO_OF_SPINORFIELDS);
  }
  else {
    j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS);
  }
  if (j != 0) {
    fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
    exit(0);
  }
  if(even_odd_flag) {
    j = init_csg_field(VOLUMEPLUSRAND/2);
  }
  else {
    j = init_csg_field(VOLUMEPLUSRAND);
  }
  if (j != 0) {
    fprintf(stderr, "Not enough memory for csg fields! Aborting...\n");
    exit(0);
  }
  j = init_moment_field(VOLUME, VOLUMEPLUSRAND + g_dbw2rand);
  if (j != 0) {
    fprintf(stderr, "Not enough memory for moment fields! Aborting...\n");
    exit(0);
  }

  if(g_running_phmc) {
    j = init_bispinor_field(VOLUME/2, NO_OF_BISPINORFIELDS);
    if (j!= 0) {
      fprintf(stderr, "Not enough memory for bi-spinor fields! Aborting...\n");
      exit(0);
    }
  }

  /*construct the filenames for the observables and the parameters*/
  strncpy(datafilename,filename,200);  
  strcat(datafilename,".data");
  strncpy(parameterfilename,filename,200);  
  strcat(parameterfilename,".para");

  /* define the geometry */
  geometry();

  /* define the boundary conditions for the fermion fields */
  boundary(g_kappa);

  status = check_geometry();

  if (status != 0) {
    fprintf(stderr, "Checking of geometry failed. Unable to proceed.\nAborting....\n");
    exit(1);
  }

#ifdef _USE_HALFSPINOR
  j = init_dirac_halfspinor();
  if (j!= 0) {
    fprintf(stderr, "Not enough memory for halffield! Aborting...\n");
    exit(-1);
  }
  if(g_sloppy_precision_flag == 1) {
    init_dirac_halfspinor32();
  }
#  if (defined _PERSISTENT)
  init_xchange_halffield();
#  endif
#endif

  /* Initialise random number generator */
  start_ranlux(rlxd_level, random_seed^trajectory_counter);

  /* Set up the gauge field */
  /* continue and restart */
  if(startoption==3 || startoption == 2) {
    if(g_proc_id == 0) {
      printf("# Trying to read gauge field from file %s in %s precision.\n",
            gauge_input_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
      fflush(stdout);
    }
    if( (status = read_gauge_field(gauge_input_filename,g_gauge_field)) != 0) {
      fprintf(stderr, "Error %d while reading gauge field from %s\nAborting...\n", status, gauge_input_filename);
      exit(-2);
    }

    if (g_proc_id == 0){
      printf("# Finished reading gauge field.\n");
      fflush(stdout);
    }
  }
  else if (startoption == 1) {
    /* hot */
    random_gauge_field(reproduce_randomnumber_flag, g_gauge_field);
  }
  else if(startoption == 0) {
    /* cold */
    unit_g_gauge_field();
  }

  /*For parallelization: exchange the gaugefield */
#ifdef MPI
  xchange_gauge(g_gauge_field);
#endif

  plaquette_energy = measure_gauge_action( (const su3**) g_gauge_field);
  if(g_rgi_C1 > 0. || g_rgi_C1 < 0.) {
    rectangle_energy = measure_rectangles( (const su3**) g_gauge_field);
    if(g_proc_id == 0){
      fprintf(parameterfile,"# Computed rectangle value: %14.12f.\n",rectangle_energy/(12.*VOLUME*g_nproc));
    }
  }

  if(g_proc_id == 0) {
    printf("# Computed plaquette value: %14.12f.\n", plaquette_energy/(6.*VOLUME*g_nproc));
  }

  /* Loop for trajectories */
  for(j = 0; j < Nmeas; j++) {
    if(g_proc_id == 0) {
      printf("#\n# Starting trajectory no %d\n", trajectory_counter);
    }

    sprintf(gauge_filename,"conf.%.4d", nstore);
    nstore ++;
    for (unsigned int attempt = 1; attempt <= io_max_attempts; ++attempt)
    {
      if (g_proc_id == 0) fprintf(stdout, "# Writing gauge field to %s.\n", tmp_filename);

      xlfInfo = construct_paramsXlfInfo(plaquette_energy/(6.*VOLUME*g_nproc), trajectory_counter);
      status = write_gauge_field( tmp_filename, gauge_precision_write_flag, xlfInfo);
      free(xlfInfo);
          
      if (status) {
        /* Writing the gauge field failed directly */
        fprintf(stderr, "Error %d while writing gauge field to %s\nAborting...\n", status, tmp_filename);
        exit(-2);
      }
          
      /* Read gauge field back to verify the writeout */
      if (g_proc_id == 0) fprintf(stdout, "# Write completed, verifying write...\n");

      for(int read_attempt = 0; read_attempt < 2; ++read_attempt) {
        status = read_gauge_field(tmp_filename,gauge_tmp);        
        if (!status) {
          if (g_proc_id == 0) fprintf(stdout, "# Write successfully verified.\n");
          break; // out of read attempt loop
        } else {
          if(g_proc_id==0) {
            if(read_attempt+1 < 2) {
              fprintf(stdout, "# Reread attempt %d out of %d failed, trying again in %d seconds!\n",read_attempt+1,2,2);
            } else {
              fprintf(stdout, "$ Reread attept %d out of %d failed, write will be reattempted!\n",read_attempt+1,2,2);
            }
          }
        sleep(2);
        }
      }

      /* we broke out of the read attempt loop, still need to break out of the write attempt loop ! */
      if(!status) {
        break;
      } 

      if (g_proc_id == 0) {
        fprintf(stdout, "# Writeout of %s returned no error, but verification discovered errors.\n", tmp_filename);
        fprintf(stdout, "# Potential disk or MPI I/O error.\n");
        fprintf(stdout, "# This was writing attempt %d out of %d.\n", attempt, io_max_attempts);
      }

      if (attempt == io_max_attempts)
        kill_with_error(NULL, g_proc_id, "Persistent I/O failures!\n");

      if (g_proc_id == 0)
        fprintf(stdout, "# Will attempt to write again in %d seconds.\n", io_timeout);
       
      sleep(io_timeout);
#ifdef MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
    }
    /* Now move .conf.tmp into place */
    if(g_proc_id == 0) {
      fprintf(stdout, "# Renaming %s to %s.\n", tmp_filename, gauge_filename);
      if (rename(tmp_filename, gauge_filename) != 0) {
        /* Errno can be inspected here for more descriptive error reporting */
        fprintf(stderr, "Error while trying to rename temporary file %s to %s. Unable to proceed.\n", tmp_filename, gauge_filename);
        exit(-2);
      }
      countfile = fopen(nstore_filename, "w");
      fprintf(countfile, "%d %d %s\n", nstore, trajectory_counter+1, gauge_filename);
      fclose(countfile);
    }

#ifdef MPI
    MPI_Barrier(MPI_COMM_WORLD);
#endif
    trajectory_counter++;
  } /* end of loop over trajectories */

#ifdef OMP
  free_omp_accumulators();
#endif
  free_gauge_tmp();
  free_gauge_field();
  free_geometry_indices();
  free_spinor_field();
  free_moment_field();
  free_monomials();
  if(g_running_phmc) {
    free_bispinor_field();
    free_chi_spinor_field();
  }
  free(input_filename);
  free(filename);
#ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return(0);
#ifdef _KOJAK_INST
#pragma pomp inst end(main)
#endif
}

static void usage(){
  fprintf(stdout, "HMC for Wilson twisted mass QCD\n");
  fprintf(stdout, "Version %s \n\n", PACKAGE_VERSION);
  fprintf(stdout, "Please send bug reports to %s\n", PACKAGE_BUGREPORT);
  fprintf(stdout, "Usage:   hmc_tm [options]\n");
  fprintf(stdout, "Options: [-f input-filename]  default: hmc.input\n");
  fprintf(stdout, "         [-o output-filename] default: output\n");
  fprintf(stdout, "         [-v] more verbosity\n");
  fprintf(stdout, "         [-V] print version information and exit\n");
  fprintf(stdout, "         [-h|-? this help]\n");
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
    *input_filename = calloc(13, sizeof(char));
    strcpy(*input_filename,"hmc.input");
  }
  
  if( *filename == NULL ) {
    *filename = calloc(7, sizeof(char));
    strcpy(*filename,"output");
  } 
}

