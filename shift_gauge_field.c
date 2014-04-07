/***********************************************************************
 *
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
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
 * invert for twisted mass QCD
 *
 * Author: Carsten Urbach
 *         urbach@physik.fu-berlin.de
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
#include "dirty_shameful_business.h"
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
#include <measurements/prepare_source.h>
#include <io/params.h>
#include <io/gauge.h>
#include <io/spinor.h>
#include <io/utils.h>
#include "solver/dirac_operator_eigenvectors.h"
#include "P_M_eta.h"
#include "operator/tm_operators.h"
#include "operator/Dov_psi.h"
#include "solver/spectral_proj.h"

extern int nstore;
int check_geometry();

static void usage();
static void process_args(int argc, char *argv[], char ** input_filename);
static void set_default_filenames(char ** input_filename);

int main(int argc, char *argv[])
{
  FILE *parameterfile = NULL;
  int j, i, ix = 0, isample = 0, op_id = 0;
  char datafilename[206];
  char parameterfilename[206];
  char conf_filename[50];
  char * input_filename = NULL;
  double plaquette_energy;
  spinor **s, *s_;

  verbose = 0;
  g_use_clover_flag = 0;

#ifdef MPI
  fatal_error("This code does not work with MPI! Aborting!\n","main");
#endif 
    
  process_args(argc,argv,&input_filename);
  set_default_filenames(&input_filename);

  /* Read the input file */
  if( (j = read_input(input_filename)) != 0) {
    fprintf(stderr, "Could not find input file: %s\nAborting...\n", input_filename);
    exit(-1);
  }

#ifdef OMP
  init_openmp();
#endif

  if (Nsave == 0) {
    Nsave = 1;
  }

  tmlqcd_mpi_init(argc, argv);

  /* Allocate needed memory */
  initialize_gauge_buffers(3);
 
  /* starts the single and double precision random number */
  /* generator                                            */
  start_ranlux(rlxd_level, random_seed);

#ifndef MPI
  g_dbw2rand = 0;
#endif

  j = init_gauge_field(VOLUMEPLUSRAND, 0);
  
  if (j != 0) {
    fprintf(stderr, "Not enough memory for gauge_fields! Aborting...\n");
    exit(-1);
  }
  j = init_geometry_indices(VOLUMEPLUSRAND);
  if (j != 0) {
    fprintf(stderr, "Not enough memory for geometry indices! Aborting...\n");
    exit(-1);
  }
  g_mu = g_mu1;

  /* define the geometry */
  geometry();

  /* define the boundary conditions for the fermion fields */
  boundary(g_kappa);

  // get a field buffer for a gauge field
  gauge_field_t tmp_gauge = get_gauge_field();

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
      printf("# The computed ORIGINAL plaquette value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
      fflush(stdout);
    }

    // shift gauge field by shifts defined in the input file
    unsigned int shifted_index;
    unsigned int original_index;
    su3 *orig, *shifted;
    for(unsigned int t=0; t<T; ++t){
      for(unsigned int x=0; x<LX; ++x){
        for(unsigned int y=0; y<LY; ++y){
          for(unsigned int z=0; z<LZ; ++z){
            original_index = Index(t,x,y,z); 
            shifted_index = Index( abs( (t+g_t_shift)%T ) ,
                                   abs( (x+g_x_shift)%LX ) ,
                                   abs( (y+g_y_shift)%LY ) ,
                                   abs( (z+g_z_shift)%LZ ) );
            for(unsigned int mu=0; mu<4; mu++){
              shifted = &tmp_gauge[shifted_index][mu];
              orig = &g_gf[original_index][mu];
              _su3_assign(*shifted,*orig);
            }
          }
        }
      }
    }

    // this remaps the g_gauge_field pointer to the tmp_gauge field buffer
    // so we can write it out in the next step (stupid hard-coded IO functions...)
    ohnohack_remap_g_gauge_field(tmp_gauge);
    
    /*compute the energy of the gauge field*/
    plaquette_energy = measure_gauge_action(_AS_GAUGE_FIELD_T(g_gauge_field));
    if (g_cart_id == 0) {
      printf("# The computed SHIFTED plaquette value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
      fflush(stdout);
    }

    char o_filename[300];
    snprintf(o_filename,300,"shifted.conf.%04d", nstore );
    paramsXlfInfo *xlfInfo = construct_paramsXlfInfo(plaquette_energy/(6.*VOLUME*g_nproc), nstore );
    int status = write_gauge_field( o_filename, gauge_precision_write_flag, xlfInfo);
    free(xlfInfo);

    // and then we map it back
    ohnohack_remap_g_gauge_field(g_gf);
    // increment the configuration file name identifier
    nstore += Nsave;  
  } 

  return_gauge_field(&g_gf);
  return_gauge_field(&tmp_gauge);

#ifdef MPI
  MPI_Finalize();
#endif
#ifdef OMP
  free_omp_accumulators();
#endif

  free_geometry_indices();
  finalize_gauge_buffers();

  free(input_filename);

  return(0);
}

static void usage()
{
  fprintf(stdout, "Code to write out a shifted gauge field as defined in the input file.\n");
  fprintf(stdout, "Version %s \n\n", PACKAGE_VERSION);
  fprintf(stdout, "Please send bug reports to %s\n", PACKAGE_BUGREPORT);
  fprintf(stdout, "Usage:   shift_gauge_field [options]\n");
  fprintf(stdout, "Options: [-f input-filename]\n");
  fprintf(stdout, "         [-v] more verbosity\n");
  fprintf(stdout, "         [-h|-? this help]\n");
  fprintf(stdout, "         [-V] print version information and exit\n");
  exit(0);
}

static void process_args(int argc, char *argv[], char ** input_filename) {
  int c;
  while ((c = getopt(argc, argv, "h?vVf:o:")) != -1) {
    switch (c) {
      case 'f':
        *input_filename = calloc(200, sizeof(char));
        strncpy(*input_filename, optarg, 200);
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

static void set_default_filenames(char ** input_filename) {
  if( *input_filename == NULL ) {
    *input_filename = calloc(24, sizeof(char));
    strcpy(*input_filename,"shift_gauge_field.input");
  }
}

