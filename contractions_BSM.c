/***********************************************************************
 *
 * Copyright (C) 2017 Ferenc Pittler
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
 ***********************************************************************/
#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include"lime.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#ifdef MPI
#include <mpi.h>
#endif
#include "global.h"
#include "git_hash.h"
#include "getopt.h"
#include "default_input_values.h"
#include "read_input.h"
#include "su3.h"
#include "operator/tm_operators.h"
#include "linalg_eo.h"
#include "geometry_eo.h"
#include "linalg/assign.h"
#include "operator/D_psi.h"
#include "operator/D_psi_BSM.h"
#include "operator/D_psi_BSM2b.h"
#include "operator/D_psi_BSM2f.h"
#include "operator/D_psi_BSM2m.h"
#include "operator/Dov_psi.h"
#include "operator/tm_operators_nd.h"
#include "operator/Hopping_Matrix.h"
#include "invert_eo.h"
#include "invert_doublet_eo.h"
#include "invert_overlap.h"
#include "invert_clover_eo.h"
#include "init/init_scalar_field.h"
#include "init/init_bsm_2hop_lookup.h"
#include "boundary.h"
#include "start.h"
#include "solver/solver.h"
#include "xchange/xchange_gauge.h"
#include "prepare_source.h"
#include <io/params.h>
#include <io/gauge.h>
#include <io/spinor.h>
#include <io/utils.h>
#include "io/scalar.h"
#include "buffers/utils_nonblocking.h"
#include "buffers/utils_nogauge.h"
#include "test/overlaptests.h"
#include "solver/index_jd.h"
#include "operator/clovertm_operators.h"
#include "operator/clover_leaf.h"
#include "operator.h"
#include "gettime.h"
#include "measure_gauge_action.h"
#include "mpi_init.h"
#include "init/init_geometry_indices.h"
#include "init/init_openmp.h"
#include "init/init_gauge_field.h"
#include "init/init_spinor_field.h"
#include "init/init_bispinor_field.h"
#include "contractions/contractions_checks.h"
#include "contractions/contractions_FP.h"
#include "solver/solver_field.h"
#include "source_generation.h"
#include "ranlxd.h"
#define TYPE_A 1
#define TYPE_B 0

#define TYPE_1 1
#define TYPE_2 0
#define TYPE_3 2
#define TYPE_4 3

#define TYPE_I 1
#define TYPE_II 0
static void usage()
{
  fprintf(stdout, "Options: [-f input-filename]\n");
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
//          fprintf(stdout,"%s %s\n",PACKAGE_STRING,git_hash);
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
    strcpy(*input_filename,"invert.input");
  }

  if( *filename == NULL ) {
    *filename = calloc(7, sizeof(char));
    strcpy(*filename,"output");
  }
}
extern int nstore;
int check_geometry();
static void set_default_filenames(char ** input_filename, char ** filename);
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename);
int main(int argc, char *argv[]){
  FILE *parameterfile = NULL;
  char datafilename[206];
  char parameterfilename[206];
  char conf_filename[50];
  char scalar_filename[50];
  char * input_filename = NULL;
  char * filename = NULL;
  double plaquette_energy;
  int i,j,isample=0,op_id=0;
  char prop_fname[200];
  char contractions_fname[200];
  int src_idx, pos;
//  int count;
  int status_geo;
  int ix;
#if defined MPI
  MPI_Status  statuses[8];
  MPI_Request *request;
  request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
#endif
  process_args(argc, argv, &input_filename,&filename);
  set_default_filenames(&input_filename, &filename);

  /* Read the input file */
  if ( (i = read_input(input_filename)) != 0)
  {
      fprintf(stderr, "Could not find input file: %s\nAborting...\n", input_filename);
      exit(-1);
  }

  if(g_proc_id==0)
  {
      fprintf(stdout, "#parameter  rho_BSM set to %f\n",  rho_BSM);
      fprintf(stdout, "#parameter  eta_BSM set to %f\n",  eta_BSM);
      fprintf(stdout, "#parameter   m0_BSM set to %f\n",   m0_BSM);
      fprintf(stdout, "#parameter mu03_BSM set to %f\n", mu03_BSM);
      fprintf(stdout, "#parameter mu01_BSM set to %f\n", mu01_BSM);
  }

#ifdef OMP
  init_openmp();
#endif
  tmlqcd_mpi_init(argc, argv);

  if(g_proc_id == 0)
  {
      fprintf(stdout,"# The number of processes is %d \n",g_nproc);
      fprintf(stdout,"# The lattice size is %d x %d x %d x %d\n",
         (int)(T*g_nproc_t), (int)(LX*g_nproc_x), (int)(LY*g_nproc_y), (int)(g_nproc_z*LZ));
      fprintf(stdout,"# The local lattice size is %d x %d x %d x %d\n",
        (int)(T), (int)(LX), (int)(LY),(int) LZ);
      fflush(stdout);
  }


  g_dbw2rand = 0;

  /* starts the single and double precision random number */
  /* generator                                            */
  start_ranlux(rlxd_level, random_seed);


#ifdef _GAUGE_COPY
  j = init_gauge_field(VOLUMEPLUSRAND, 1);
#else
  j = init_gauge_field(VOLUMEPLUSRAND, 0);
#endif

  if (j != 0)
  {
      fprintf(stderr, "Not enough memory for gauge_fields! Aborting...\n");
      exit(-1);
  }

  init_geometry_indices(VOLUMEPLUSRAND);

/* Iniiialising the spinor fields */
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
  for(j = 0; j < no_operators; j++) if(!operator_list[j].even_odd_flag) even_odd_flag = 0;

#ifndef MPI
  g_dbw2rand = 0;
#endif

  if (even_odd_flag)
  {
      j = init_spinor_field(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS);
  }
  else
  {
      j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS);
  }
  if (j != 0)
  {
      fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
      exit(-1);
  }
  j = init_bispinor_field(VOLUMEPLUSRAND, 48);
  if ( j!= 0)
  {
      fprintf(stderr, "Not enough memory for bispinor fields! Aborting...\n");
      exit(0);
  }

  int numbScalarFields = 4;
  j = init_scalar_field(VOLUMEPLUSRAND, numbScalarFields);
  if ( j!= 0)
  {
      fprintf(stderr, "Not enough memory for scalar fields! Aborting...\n");
      exit(0);
  }

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field, VOLUMEPLUSRAND, 2);


  /* define the geometry */

  geometry();
  g_kappa=-1;
  if ((g_cart_id == 0) && (g_kappa != -1))
  {
      fprintf(stdout, "#error anti-periodic boundary condition is implemented via g_kappa %e\n",g_kappa);
      exit(1);
  }
  boundary(g_kappa);

  status_geo = check_geometry();
  if (status_geo != 0)
  {
      fprintf(stderr, "Checking of geometry failed. Unable to proceed.\nAborting....\n");
      exit(1);
  }


  if (Nsave == 0) {
    Nsave = 1;
  }

  g_mu = g_mu1;

  if (g_cart_id == 0)
  {
    /*construct the filenames for the observables and the parameters*/
      strncpy(datafilename, filename, 200);
      strcat(datafilename, ".data");
      strncpy(parameterfilename, filename, 200);
      strcat(parameterfilename, ".para");

      parameterfile = fopen(parameterfilename, "w");
      write_first_messages(parameterfile, "invert", git_hash);
      fclose(parameterfile);
  }

  init_operators();


  for (j = 0; j < Nmeas; j++)
  {
      sprintf(conf_filename, "%s.%.4d", gauge_input_filename, nstore);
      if (g_cart_id == 0)
      {
          printf("#\n# Trying to read gauge field from file %s in %s precision.\n",
            conf_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
          fflush(stdout);
      }
      if ( (i = read_gauge_field(conf_filename,g_gauge_field) ) !=0)
      {
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
      plaquette_energy = measure_plaquette( (const su3**) g_gauge_field);


      if (g_cart_id == 0) {
          printf("# The computed plaquette value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
          fflush(stdout);
      }
      if(SourceInfo.type == 1) {
          index_start = 0;
          index_end = 1;
      }


      if (g_cart_id == 0) {
          fprintf(stdout, "#\n"); /*Indicate starting of the operator part*/
      }
      for (op_id =0; op_id < no_operators; op_id++){
          if (operator_list[op_id].type== BSM2f){
              init_D_psi_BSM2f();
          }
          boundary( operator_list[op_id].kappa);
          g_kappa = operator_list[op_id].kappa;
          if (g_cart_id ==0) {fprintf(stdout, "#kappa value=%e\n", g_kappa);}
          g_mu = 0.;
          if (g_cart_id == 0) printf("# npergauge=%d\n", operator_list[op_id].npergauge);

          if (g_cart_id == 0) printf("# Starting scalar counter is %d for gauge field %d \n", nscalar, nstore );
          /* support multiple inversions for the BSM operator, one for each scalar field */

          for(int i_pergauge = 0; i_pergauge < operator_list[op_id].npergauge; ++i_pergauge){
             /* set scalar field counter to InitialScalarCounter */
             int iscalar = nscalar+j*operator_list[op_id].nscalarstep*operator_list[op_id].npergauge+i_pergauge*operator_list[op_id].nscalarstep;
             operator_list[op_id].n = iscalar;
          // read scalar field
             if( strcmp(scalar_input_filename, "create_random_scalarfield") == 0 )
             {
                for( int s = 0; s < 4; s++) { ranlxd(g_scalar_field[s], VOLUME); }
             }
             else
             {
                snprintf(scalar_filename, 50, "%s.%.8d", scalar_input_filename, iscalar);
                if (g_cart_id == 0)
                {
                    printf("#\n# Trying to read scalar field from file %s in %s precision.\n",
                       scalar_filename, (scalar_precision_read_flag == 32 ? "single" : "double"));
                    fflush(stdout);
                }
                int i;
                double read_end, read_begin=gettime();

                if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0)
                {
                    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
                    exit(-2);
                }
                read_end=gettime();

                if (g_cart_id == 0) {
                   printf("# Finished reading scalar field in %.4e seconds.\n",read_end-read_begin); 
                   fflush(stdout);
                }

             }//End of reading scalar field

//             unit_scalar_field(g_scalar_field);
#if defined MPI
             for( int s=0; s<4; s++ )
                generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
#endif
             for( isample = 0; isample < no_samples; isample++)
             {
               if (propagatorsonthefly_BSM == 1){

                 if ((g_cart_id == 0 ) && ( (index_start != 0) || (index_end!= 12) ))
                 {
                    fprintf(stderr, "Contraction can be computed only with full set of point propagators\n");
                    exit(1);
                 }

                 for(ix = index_start; ix < index_end; ix++) {
                    if (g_cart_id == 0) {
                      fprintf(stdout, "#\n"); /*Indicate starting of new index*/
                    }

                    /* we use g_spinor_field[0-7] for sources and props for the moment */
                    /* 0-3 in case of 1 flavour  */
                    /* 0-7 in case of 2 flavours */

                    prepare_source(nstore, isample, ix, op_id, read_source_flag, source_location);

//                  if (g_cart_id == 0) printf("Source has been prepared\n\n\n");
                    //randmize initial guess for eigcg if needed-----experimental
                    if( (operator_list[op_id].solver == INCREIGCG) && (operator_list[op_id].solver_params.eigcg_rand_guess_opt) )
                    { //randomize the initial guess
                        gaussian_volume_source( operator_list[op_id].prop0, operator_list[op_id].prop1,isample,ix,0); //need to check this
                    } 
		    
		    operator_list[op_id].inverter_save(op_id, index_start, 1);

                 }//end of loop for spinor and color source degrees of freedom
               }
               else{
                 for(src_idx = 0; src_idx < 12; src_idx++ )
                 {
                    snprintf(prop_fname,200,"bsm2prop.%.4d.%.2d.%02d.%.8d.inverted",nstore, isample, src_idx, iscalar);

                    for(pos = 0; pos < 8; ){
                       printf("READCHECK: Propagator in pos %02d from file %s\n", pos/2,prop_fname);

//read the propagator from source d to sink d 
                       read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
                       convert_eo_to_lexic(temp_field[0], g_spinor_field[0], g_spinor_field[1]);
                       pos+=1;

//read the propagator from source d to sink u
                       read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
                       convert_eo_to_lexic(temp_field[1], g_spinor_field[0], g_spinor_field[1]);
                       pos+=1;

//create a bispinor first insert sink u then sink d
//Store them in such a way that the u-ones should come first
                       compact(g_bispinor_field[pos > 4 ? src_idx*4+pos/2-3 : src_idx*4+pos/2+ 1], temp_field[1], temp_field[0]);
                    }

                 }//end of loop for spinor and color source degrees of freedom
               }
               if (g_cart_id == 0){
                    snprintf(contractions_fname,200,"bsmcontractions.%.4d.%d.%.8d",nstore, src_idx, iscalar);
               }

               if (smearedcorrelator_BSM == 1){
                smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field);
                if (g_cart_id == 0) printf("Smeared : %e\t Non smeared %e\n", g_smeared_scalar_field[0][0],g_scalar_field[0][0]);
                 for ( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
               }
               if (g_cart_id == 0) {
                 printf("Following measurements will be done\n");
                 if (densitydensity_BSM == 1) printf("#Density Density correlation function\n");
                 if (densitydensity_s0s0_BSM == 1) printf("#Density Density s0s0-p0p0 using trivial scalar field\n");
                 if (densitydensity_sxsx_BSM == 1) printf("#Density Density sxsx-pxpx using trivial scalar field\n");
                 if (diraccurrentdensity_BSM == 1) printf("#Dirac current density correlation function\n");
                 if (wilsoncurrentdensitypr1_BSM == 1) printf("#Wilson  current density PR1 correlation function\n");
                 if (wilsoncurrentdensitypr2_BSM == 1) printf("#Wilson  current density PR2 correlation function\n");
                 if (wilsoncurrentdensitypl1_BSM == 1) printf("#Wilson  current density PL1 correlation function\n");
                 if (wilsoncurrentdensitypl2_BSM == 1) printf("#Wilson  current density PL2 correlation function\n");
               }

               if (densitydensity_BSM == 1){
                 density_density_1234(g_bispinor_field, TYPE_1, contractions_fname);
//                 density_density_1234_petros(g_bispinor_field);
                 density_density_1234(g_bispinor_field, TYPE_2, contractions_fname);
                 density_density_1234(g_bispinor_field, TYPE_3, contractions_fname);
                 density_density_1234(g_bispinor_field, TYPE_4, contractions_fname);
               }
               if (densitydensity_s0s0_BSM == 1){
                 unit_scalar_field(g_scalar_field);
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field);
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
                 density_density_1234_s0s0(g_bispinor_field, TYPE_1, contractions_fname);
                 density_density_1234_s0s0(g_bispinor_field, TYPE_2, contractions_fname);
                 density_density_1234_s0s0(g_bispinor_field, TYPE_3, contractions_fname);
                 density_density_1234_s0s0(g_bispinor_field, TYPE_4, contractions_fname);
                 double read_end, read_begin=gettime();
                 if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0 )
                 {
                    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
                    exit(-2);
                 }
                 read_end=gettime();
                 if (g_cart_id == 0) {
                   printf("# Finished reading scalar field in %.4e seconds.\n",read_end-read_begin);
                   fflush(stdout);
                 }
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field);
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
               } 
               if (densitydensity_sxsx_BSM ==1){
                 unit_scalar_field(g_scalar_field);
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field);
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
                 density_density_1234_sxsx(g_bispinor_field, TYPE_1, contractions_fname);
                 density_density_1234_sxsx(g_bispinor_field, TYPE_2, contractions_fname);
                 density_density_1234_sxsx(g_bispinor_field, TYPE_3, contractions_fname);
                 density_density_1234_sxsx(g_bispinor_field, TYPE_4, contractions_fname);
                 double read_end, read_begin=gettime();
                 if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0 )
                 {
                    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
                    exit(-2);
                 }
                 read_end=gettime();
                 if (g_cart_id == 0) {
                   printf("# Finished reading scalar field in %.4e seconds.\n",read_end-read_begin);
                   fflush(stdout);
                 }
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field);
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
               }
               if (diraccurrentdensity_BSM == 1){
                 naivedirac_current_density_12ab( g_bispinor_field, TYPE_I , TYPE_A, contractions_fname);
//                 diraccurrent1a_petros( g_bispinor_field );
                 naivedirac_current_density_12ab( g_bispinor_field, TYPE_I , TYPE_B, contractions_fname);
                 naivedirac_current_density_12ab( g_bispinor_field, TYPE_II, TYPE_A, contractions_fname);
                 naivedirac_current_density_12ab( g_bispinor_field, TYPE_II, TYPE_B, contractions_fname);
               }
               if (wilsoncurrentdensitypr1_BSM == 1){
                 wilsonterm_current_density_312ab( g_bispinor_field, TYPE_1, TYPE_A, contractions_fname);
//                 wilsoncurrent_density_3_petros( g_bispinor_field );
                 wilsonterm_current_density_312ab( g_bispinor_field, TYPE_1, TYPE_B, contractions_fname);
                 wilsonterm_current_density_312ab( g_bispinor_field, TYPE_2, TYPE_A, contractions_fname);
                 wilsonterm_current_density_312ab( g_bispinor_field, TYPE_2, TYPE_B, contractions_fname);
               }
               if (wilsoncurrentdensitypr2_BSM == 1){
                 wilsonterm_current_density_412ab( g_bispinor_field, TYPE_1, TYPE_A, contractions_fname);
                 wilsonterm_current_density_412ab( g_bispinor_field, TYPE_1, TYPE_B, contractions_fname);
                 wilsonterm_current_density_412ab( g_bispinor_field, TYPE_2, TYPE_A, contractions_fname);
                 wilsonterm_current_density_412ab( g_bispinor_field, TYPE_2, TYPE_B, contractions_fname);
               }
               if (wilsoncurrentdensitypl1_BSM == 1){
                 wilsonterm_current_density_512ab( g_bispinor_field, TYPE_1, TYPE_A, contractions_fname);
                 wilsonterm_current_density_512ab( g_bispinor_field, TYPE_1, TYPE_B, contractions_fname);
                 wilsonterm_current_density_512ab( g_bispinor_field, TYPE_2, TYPE_A, contractions_fname);
                 wilsonterm_current_density_512ab( g_bispinor_field, TYPE_2, TYPE_B, contractions_fname);
               }
               if (wilsoncurrentdensitypl2_BSM == 1){
               //wilsoncurrent61a_petros( g_bispinor_field );
                 wilsonterm_current_density_612ab( g_bispinor_field, TYPE_1, TYPE_A, contractions_fname);
                 wilsonterm_current_density_612ab( g_bispinor_field, TYPE_1, TYPE_B, contractions_fname);
               //wilsoncurrent62a_petros( g_bispinor_field );
                 wilsonterm_current_density_612ab( g_bispinor_field, TYPE_2, TYPE_A, contractions_fname);
                 wilsonterm_current_density_612ab( g_bispinor_field, TYPE_2, TYPE_B, contractions_fname);
               }
//               density_density_1234_petros(g_bispinor_field);

             } //End of loop over samples

          } //End loop over scalar fields
          if (operator_list[op_id].type == BSM2f ){
             free_D_psi_BSM2f();
          }

      }//End loop over operators

      nstore+=Nsave;
  }//End of loop over gauges

  finalize_solver(temp_field,2);
#if defined MPI
  free(request);
#endif
  free_gauge_field();
  free_geometry_indices();
  free_bispinor_field();
  free_scalar_field();
  free_spinor_field();
#if defined MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif

}
