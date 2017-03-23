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
#include "solver/solver_field.h"
#include "ranlxd.h"
#define DAGGER 1
#define NO_DAGG 0 

#define GAMMA_UP 1
#define GAMMA_DN -1
#define NO_GAMMA 0

#define WITH_SCALAR 1
#define NO_SCALAR 0

#define TYPE_A 1
#define TYPE_B 0

#define TYPE_1 1
#define TYPE_2 0
#define TYPE_3 2
#define TYPE_4 3

#define TYPE_I 1
#define TYPE_II 0
/* indexing of propfields;
   
   propagator for  (dagger or nondagger source)
              for  flavor component f
              for  color  component c    
              for  spinor component s
   is the following bispinor array of size VOLUME(PLUSRAND)

   propfields[12*s + 4*c + 2*f + dagg ? 1: 0]  
     
 */
/**************************
Multiplication with the backward propagator

S == matrix element of D^-1 between the following states

S( ytilde , x+-dir )       psi   x
   flavor2, flavor1    x         flavor1
   spinor2, spinor1              spinor1
   color 2, color 1              color1

=
Stilde* (x+-dir , ytilde)      psi   x
         flavor1, flavor2  x         flavor1
         spinor1, spinor2            spinor1  
         color 1, color 2            color1
where Stilde is the matrix element of D^dagger^-1 between 
the correspondig states

**************************/
static _Complex double bispinor_scalar_product ( bispinor *s1, bispinor *s2 ){
   _Complex double res=0.0;
   res   =s2->sp_up.s0.c0 * conj(s1->sp_up.s0.c0) + s2->sp_up.s0.c1 * conj(s1->sp_up.s0.c1) + s2->sp_up.s0.c2 * conj(s1->sp_up.s0.c2) +
          s2->sp_up.s1.c0 * conj(s1->sp_up.s1.c0) + s2->sp_up.s1.c1 * conj(s1->sp_up.s1.c1) + s2->sp_up.s1.c2 * conj(s1->sp_up.s1.c2) +
          s2->sp_up.s2.c0 * conj(s1->sp_up.s2.c0) + s2->sp_up.s2.c1 * conj(s1->sp_up.s2.c1) + s2->sp_up.s2.c2 * conj(s1->sp_up.s2.c2) +
          s2->sp_up.s3.c0 * conj(s1->sp_up.s3.c0) + s2->sp_up.s3.c1 * conj(s1->sp_up.s3.c1) + s2->sp_up.s3.c2 * conj(s1->sp_up.s3.c2) +
          s2->sp_dn.s0.c0 * conj(s1->sp_dn.s0.c0) + s2->sp_dn.s0.c1 * conj(s1->sp_dn.s0.c1) + s2->sp_dn.s0.c2 * conj(s1->sp_dn.s0.c2) +
          s2->sp_dn.s1.c0 * conj(s1->sp_dn.s1.c0) + s2->sp_dn.s1.c1 * conj(s1->sp_dn.s1.c1) + s2->sp_dn.s1.c2 * conj(s1->sp_dn.s1.c2) +
          s2->sp_dn.s2.c0 * conj(s1->sp_dn.s2.c0) + s2->sp_dn.s2.c1 * conj(s1->sp_dn.s2.c1) + s2->sp_dn.s2.c2 * conj(s1->sp_dn.s2.c2) +
          s2->sp_dn.s3.c0 * conj(s1->sp_dn.s3.c0) + s2->sp_dn.s3.c1 * conj(s1->sp_dn.s3.c1) + s2->sp_dn.s3.c2 * conj(s1->sp_dn.s3.c2);
   return res;
}
static void multiply_backward_propagator( bispinor *dest, bispinor **propagator, bispinor *source, int idx, int dir){
   int propcoord;
   bispinor source_copy;
   if (dir == NODIR){
      propcoord=idx;
   } 
   else if (dir == TUP){
      propcoord=g_iup[idx][TUP];
   }
   else if (dir == TDOWN){
      propcoord=g_idn[idx][TUP];
   }
   else{
      propcoord=0;
      if (g_cart_id == 0){ fprintf(stderr,"Wrong direction in multiply backward prop\n"); 
                           exit(1); }
   }

   _spinor_assign( source_copy.sp_dn, source->sp_dn);
   _spinor_assign( source_copy.sp_up, source->sp_up);

   dest->sp_up.s0.c0= bispinor_scalar_product ( &propagator[ 1][propcoord], &source_copy );
   dest->sp_up.s0.c1= bispinor_scalar_product ( &propagator[ 5][propcoord], &source_copy );
   dest->sp_up.s0.c2= bispinor_scalar_product ( &propagator[ 9][propcoord], &source_copy );

   dest->sp_up.s1.c0= bispinor_scalar_product ( &propagator[13][propcoord], &source_copy );
   dest->sp_up.s1.c1= bispinor_scalar_product ( &propagator[17][propcoord], &source_copy );
   dest->sp_up.s1.c2= bispinor_scalar_product ( &propagator[21][propcoord], &source_copy );

   dest->sp_up.s2.c0= bispinor_scalar_product ( &propagator[25][propcoord], &source_copy );
   dest->sp_up.s2.c1= bispinor_scalar_product ( &propagator[29][propcoord], &source_copy );
   dest->sp_up.s2.c2= bispinor_scalar_product ( &propagator[33][propcoord], &source_copy );

   dest->sp_up.s3.c0= bispinor_scalar_product ( &propagator[37][propcoord], &source_copy );
   dest->sp_up.s3.c1= bispinor_scalar_product ( &propagator[41][propcoord], &source_copy );
   dest->sp_up.s3.c2= bispinor_scalar_product ( &propagator[45][propcoord], &source_copy );

   dest->sp_dn.s0.c0= bispinor_scalar_product ( &propagator[ 3][propcoord], &source_copy );
   dest->sp_dn.s0.c1= bispinor_scalar_product ( &propagator[ 7][propcoord], &source_copy );
   dest->sp_dn.s0.c2= bispinor_scalar_product ( &propagator[11][propcoord], &source_copy );

   dest->sp_dn.s1.c0= bispinor_scalar_product ( &propagator[15][propcoord], &source_copy );
   dest->sp_dn.s1.c1= bispinor_scalar_product ( &propagator[19][propcoord], &source_copy );
   dest->sp_dn.s1.c2= bispinor_scalar_product ( &propagator[23][propcoord], &source_copy );

   dest->sp_dn.s2.c0= bispinor_scalar_product ( &propagator[27][propcoord], &source_copy );
   dest->sp_dn.s2.c1= bispinor_scalar_product ( &propagator[31][propcoord], &source_copy );
   dest->sp_dn.s2.c2= bispinor_scalar_product ( &propagator[35][propcoord], &source_copy );

   dest->sp_dn.s3.c0= bispinor_scalar_product ( &propagator[39][propcoord], &source_copy );
   dest->sp_dn.s3.c1= bispinor_scalar_product ( &propagator[43][propcoord], &source_copy );
   dest->sp_dn.s3.c2= bispinor_scalar_product ( &propagator[47][propcoord], &source_copy );
}
//dest used as a source, an output it is overwritten
static void taui_scalarfield_flavoronly( _Complex double *dest, int tauindex, int dagger ){
   _Complex double *source_copy;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;
   int i;
  
   source_copy=(double *)malloc(sizeof(double)*4*T_global);

   if (source_copy == NULL) {
      if (g_cart_id == 0) {printf("memory allocation failed\n"); exit(1);}
   }
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];
   
   if (dagger == DAGGER){
     if (tauindex == 0){
       if (smearedcorrelator_BSM == 1){
         a11=  -1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
         a12=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];

         a21=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
         a22=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
       }
       else{
         a11=  -1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
         a12=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];

         a21=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
         a22=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
       }
     }
     else  if (tauindex == 1){
       if (smearedcorrelator_BSM == 1) {
         a11=  +1.*g_smeared_scalar_field[1][0] - I*g_smeared_scalar_field[2][0];
         a12=  -1.*g_smeared_scalar_field[3][0] - I*g_smeared_scalar_field[0][0];

         a21=  -1.*g_smeared_scalar_field[3][0] + I*g_smeared_scalar_field[0][0];
         a22=  -1.*g_smeared_scalar_field[1][0] - I*g_smeared_scalar_field[2][0];
       }
       else{
         a11=  +1.*g_scalar_field[1][0] - I*g_scalar_field[2][0];
         a12=  -1.*g_scalar_field[3][0] - I*g_scalar_field[0][0];

         a21=  -1.*g_scalar_field[3][0] + I*g_scalar_field[0][0];
         a22=  -1.*g_scalar_field[1][0] - I*g_scalar_field[2][0];
       }
     }
     else  if (tauindex == 2){
       if (smearedcorrelator_BSM == 1){ 
         a11=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
         a12=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];

         a21=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
         a22=  -1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
       }
       else{
         a11=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
         a12=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];

         a21=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
         a22=  -1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
       }
     }
   }
   else if (dagger == NO_DAGG){
     if (tauindex == 0){
       if (smearedcorrelator_BSM == 1){
         a11=  -1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
         a12=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];

         a21=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
         a22=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
       }
       else{
         a11=  -1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
         a12=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];

         a21=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
         a22=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
       }
     }
     else if (tauindex == 1){
      if (smearedcorrelator_BSM == 1){
         a11=  +1.*g_smeared_scalar_field[1][0] + I*g_smeared_scalar_field[2][0];
         a12=  -1.*g_smeared_scalar_field[3][0] - I*g_smeared_scalar_field[0][0];

         a21=  -1.*g_smeared_scalar_field[3][0] + I*g_smeared_scalar_field[0][0];
         a22=  -1.*g_smeared_scalar_field[1][0] + I*g_smeared_scalar_field[2][0];
      }
      else{
         a11=  +1.*g_scalar_field[1][0] + I*g_scalar_field[2][0];
         a12=  -1.*g_scalar_field[3][0] - I*g_scalar_field[0][0];

         a21=  -1.*g_scalar_field[3][0] + I*g_scalar_field[0][0];
         a22=  -1.*g_scalar_field[1][0] + I*g_scalar_field[2][0];
      }
     }
     else if (tauindex == 2){
      if (smearedcorrelator_BSM == 1){
         a11=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
         a12=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];

         a21=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
         a22=  -1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
      }
      else{
         a11=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
         a12=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];

         a21=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
         a22=  -1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
      }
     }
     else{
      a11=0.;
      a12=0.;
      a21=0.;
      a22=0.;
      if (g_cart_id == 0){printf("Wrong Pauli matrix index\n");exit(1);};
    }
   }
   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);  
}
//dest used as a source, an output it is overwritten
static void taui_scalarfield_flavoronly_s0s0( _Complex double *dest, int dagger ){
   _Complex double *source_copy;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;
   int i;
 
   source_copy=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];
   if (dagger == DAGGER){
     if (smearedcorrelator_BSM == 1){
       a11=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
       a12=  -1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];

       a21=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
       a22=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
     }
     else{
       a11=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
       a12=  -1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];

       a21=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
       a22=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];     
     }
   }
   else if (dagger==NO_DAGG){
     if (smearedcorrelator_BSM == 1){
       a11=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
       a12=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];

       a21=  -1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
       a22=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
     }
     else{
       a11=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
       a12=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];

       a21=  -1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
       a22=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
     }
   }
   else{
      a11=0.;
      a12=0.;
      a21=0.;
      a22=0.;
      if (g_cart_id == 0){printf("Wrong Dagger index\n"); exit(1);}
   }
   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);
}
static void taui_spinor( bispinor *dest, bispinor *source, int tauindex ){

   su3_vector tmp2;
   bispinor tmp;
   _spinor_assign(tmp.sp_up, source->sp_up);
   _spinor_assign(tmp.sp_dn, source->sp_dn);

   if (tauindex == 0 ){
    _vector_assign(tmp2        , tmp.sp_up.s0);
    _vector_assign(tmp.sp_up.s0, tmp.sp_dn.s0);
    _vector_assign(tmp.sp_dn.s0, tmp2);

    _vector_assign(tmp2        , tmp.sp_up.s1);
    _vector_assign(tmp.sp_up.s1, tmp.sp_dn.s1);
    _vector_assign(tmp.sp_dn.s1, tmp2);
 
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
   else if (tauindex == 1 ){
    _vector_assign(tmp2             ,tmp.sp_up.s0);
    _vector_i_mul( tmp.sp_up.s0, -1 ,tmp.sp_dn.s0);
    _vector_i_mul( tmp.sp_dn.s0, +1 ,tmp2);

    _vector_assign(tmp2             ,tmp.sp_up.s1);
    _vector_i_mul( tmp.sp_up.s1, -1 ,tmp.sp_dn.s1);
    _vector_i_mul( tmp.sp_dn.s1, +1, tmp2);   
     
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
   else if (tauindex == 2 ){
    _vector_mul(tmp.sp_dn.s0, -1, tmp.sp_dn.s0);
    _vector_mul(tmp.sp_dn.s1, -1, tmp.sp_dn.s1);
     
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
}

static void taui_scalarfield_spinor_s0s0( bispinor *dest, bispinor *source, int gamma5, int idx, int direction, int dagger){

  bispinor tmp;
  bispinor tmpbi2;
  _Complex double a11=0., a12=0., a21=0., a22=0.;

  int scalarcoord;

  _spinor_assign(tmp.sp_up, source->sp_up);
  _spinor_assign(tmp.sp_dn, source->sp_dn);

 if (direction == NODIR)
   scalarcoord=idx;
 else if (direction<4){
   scalarcoord= g_iup[idx][direction];
 }
 else if (direction<8){
   scalarcoord= g_idn[idx][7-direction];
 }
 else{
   scalarcoord=0;
   if (g_cart_id == 0) {printf("Wrong direction in tau scalar field spinor\n"); exit(1);}
 }
 if (dagger == DAGGER){
   if (smearedcorrelator_BSM == 1){
     a11=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];
     a12=  -1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];
     a22=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];

   }
   else{
     a11=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
     a12=  -1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];

     a21=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
     a22=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
   }
 }
 else if (dagger == NO_DAGG){
   if (smearedcorrelator_BSM == 1){
     a11=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];

     a21=  -1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];
     a22=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];

   }
   else{
     a11=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
     a12=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];

     a21=  -1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];
     a22=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
   }
 }
 else {
   fprintf(stdout, "The sixth argument must be either DAGGER or NO_DAGG\n");
 }
 _spinor_null(tmpbi2.sp_up);
 _spinor_null(tmpbi2.sp_dn);

 if ( gamma5 == GAMMA_UP){
  _vector_mul_complex(    tmpbi2.sp_up.s0, a11, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_up.s0, a12, tmp.sp_dn.s0);

  _vector_mul_complex    (tmpbi2.sp_dn.s0, a21, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_dn.s0, a22, tmp.sp_dn.s0);

  _vector_mul_complex(    tmpbi2.sp_up.s1, a11, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_up.s1, a12, tmp.sp_dn.s1);

  _vector_mul_complex    (tmpbi2.sp_dn.s1, a21, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_dn.s1, a22, tmp.sp_dn.s1);
 }
 else if  ( gamma5 == GAMMA_DN ){
  _vector_mul_complex(    tmpbi2.sp_up.s2, a11, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_up.s2, a12, tmp.sp_dn.s2);

  _vector_mul_complex    (tmpbi2.sp_dn.s2, a21, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_dn.s2, a22, tmp.sp_dn.s2);

  _vector_mul_complex(    tmpbi2.sp_up.s3, a11, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_up.s3, a12, tmp.sp_dn.s3);

  _vector_mul_complex    (tmpbi2.sp_dn.s3, a21, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_dn.s3, a22, tmp.sp_dn.s3);
 }
 else if ( gamma5 == NO_GAMMA ){
  _spinor_mul_complex    (tmpbi2.sp_up,    a11, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_up,    a12, tmp.sp_dn);

  _spinor_mul_complex    (tmpbi2.sp_dn,    a21, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_dn,    a22, tmp.sp_dn);
 }

 _spinor_assign(dest->sp_up, tmpbi2.sp_up);
 _spinor_assign(dest->sp_dn, tmpbi2.sp_dn);

}
static void taui_scalarfield_spinor( bispinor *dest, bispinor *source, int gamma5, int tauindex, int idx, int direction, int dagger){
    
  bispinor tmp;
  bispinor tmpbi2;
  _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;

  int scalarcoord;

  _spinor_assign(tmp.sp_up, source->sp_up);
  _spinor_assign(tmp.sp_dn, source->sp_dn);

 if (direction == NODIR)
   scalarcoord=idx;
 else if (direction == TUP ){
   scalarcoord= g_iup[idx][TUP];
 }
 else if (direction == TDOWN){
   scalarcoord= g_idn[idx][TUP];
 }
 else{
   scalarcoord=0;
   if (g_cart_id == 0) {printf("Wrong direction in tau scalar field spinor\n"); exit(1);}
 }
 if (dagger == DAGGER){
  if (tauindex == 0){
   if (smearedcorrelator_BSM  == 1){
     a11=  -1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
     a22=  +1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];

   }
   else{
     a11=  -1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
     a12=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];

     a21=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
     a22=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
   }
  }
  else  if (tauindex == 1){
   if (smearedcorrelator_BSM  == 1){
     a11=  +1.*g_smeared_scalar_field[1][scalarcoord] - I*g_smeared_scalar_field[2][scalarcoord];
     a12=  -1.*g_smeared_scalar_field[3][scalarcoord] - I*g_smeared_scalar_field[0][scalarcoord];

     a21=  -1.*g_smeared_scalar_field[3][scalarcoord] + I*g_smeared_scalar_field[0][scalarcoord];
     a22=  -1.*g_smeared_scalar_field[1][scalarcoord] - I*g_smeared_scalar_field[2][scalarcoord];

   }
   else{
     a11=  +1.*g_scalar_field[1][scalarcoord] - I*g_scalar_field[2][scalarcoord];
     a12=  -1.*g_scalar_field[3][scalarcoord] - I*g_scalar_field[0][scalarcoord];

     a21=  -1.*g_scalar_field[3][scalarcoord] + I*g_scalar_field[0][scalarcoord];
     a22=  -1.*g_scalar_field[1][scalarcoord] - I*g_scalar_field[2][scalarcoord];
   }
  }
  else  if (tauindex == 2){
   if (smearedcorrelator_BSM  == 1){
     a11=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];
     a22=  -1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];

   }
   else{
     a11=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
     a12=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];

     a21=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
     a22=  -1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
   }
  }
 }
 else if (dagger == NO_DAGG){
  if (tauindex == 0){
   if (smearedcorrelator_BSM  == 1){
     a11=  -1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
     a22=  +1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];

   }
   else{
     a11=  -1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];
     a12=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];

     a21=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
     a22=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];
   }
  }
  else if (tauindex == 1){
   if (smearedcorrelator_BSM  == 1){
     a11=  +1.*g_smeared_scalar_field[1][scalarcoord] + I*g_smeared_scalar_field[2][scalarcoord];
     a12=  -1.*g_smeared_scalar_field[3][scalarcoord] - I*g_smeared_scalar_field[0][scalarcoord];

     a21=  -1.*g_smeared_scalar_field[3][scalarcoord] + I*g_smeared_scalar_field[0][scalarcoord];
     a22=  -1.*g_smeared_scalar_field[1][scalarcoord] + I*g_smeared_scalar_field[2][scalarcoord];
   }
   else{
     a11=  +1.*g_scalar_field[1][scalarcoord] + I*g_scalar_field[2][scalarcoord];
     a12=  -1.*g_scalar_field[3][scalarcoord] - I*g_scalar_field[0][scalarcoord];

     a21=  -1.*g_scalar_field[3][scalarcoord] + I*g_scalar_field[0][scalarcoord];
     a22=  -1.*g_scalar_field[1][scalarcoord] + I*g_scalar_field[2][scalarcoord];
   }
  }
  else if (tauindex == 2){
   if (smearedcorrelator_BSM  == 1){
     a11=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];
     a22=  -1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
   }
   else{
     a11=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
     a12=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];

     a21=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
     a22=  -1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
   }
  }
 }
 _spinor_null(tmpbi2.sp_up);
 _spinor_null(tmpbi2.sp_dn);
 
 if ( gamma5 == GAMMA_UP){
  _vector_mul_complex(    tmpbi2.sp_up.s0, a11, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_up.s0, a12, tmp.sp_dn.s0);

  _vector_mul_complex    (tmpbi2.sp_dn.s0, a21, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_dn.s0, a22, tmp.sp_dn.s0);

  _vector_mul_complex(    tmpbi2.sp_up.s1, a11, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_up.s1, a12, tmp.sp_dn.s1);

  _vector_mul_complex    (tmpbi2.sp_dn.s1, a21, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_dn.s1, a22, tmp.sp_dn.s1);
 }
 else if  ( gamma5 == GAMMA_DN ){
  _vector_mul_complex(    tmpbi2.sp_up.s2, a11, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_up.s2, a12, tmp.sp_dn.s2);

  _vector_mul_complex    (tmpbi2.sp_dn.s2, a21, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_dn.s2, a22, tmp.sp_dn.s2);

  _vector_mul_complex(    tmpbi2.sp_up.s3, a11, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_up.s3, a12, tmp.sp_dn.s3);

  _vector_mul_complex    (tmpbi2.sp_dn.s3, a21, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_dn.s3, a22, tmp.sp_dn.s3);
 }
 else if ( gamma5 == NO_GAMMA ){
  _spinor_mul_complex    (tmpbi2.sp_up,    a11, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_up,    a12, tmp.sp_dn);

  _spinor_mul_complex    (tmpbi2.sp_dn,    a21, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_dn,    a22, tmp.sp_dn);
 }

 _spinor_assign(dest->sp_up, tmpbi2.sp_up);
 _spinor_assign(dest->sp_dn, tmpbi2.sp_dn);

}
static void trace_in_spinor( _Complex double *dest, _Complex double *src, int spinorindex){
   int tind, find;
   for (tind=0; tind<T_global; ++tind)
     for (find=0; find<2; ++find){ 
       dest[2*tind+find]+=src[8*tind+4*find+spinorindex];
     }
}
static void trace_in_color(_Complex double *dest, bispinor *src, int colorindex){
   if      ( colorindex == 0 ){
     dest[0]+= src->sp_up.s0.c0;
     dest[1]+= src->sp_up.s1.c0;
     dest[2]+= src->sp_up.s2.c0;
     dest[3]+= src->sp_up.s3.c0;
     dest[4]+= src->sp_dn.s0.c0;
     dest[5]+= src->sp_dn.s1.c0;
     dest[6]+= src->sp_dn.s2.c0;
     dest[7]+= src->sp_dn.s3.c0;

   }
   else if ( colorindex == 1 ){
     dest[0]+= src->sp_up.s0.c1;
     dest[1]+= src->sp_up.s1.c1;
     dest[2]+= src->sp_up.s2.c1;
     dest[3]+= src->sp_up.s3.c1;
     dest[4]+= src->sp_dn.s0.c1;
     dest[5]+= src->sp_dn.s1.c1;
     dest[6]+= src->sp_dn.s2.c1;
     dest[7]+= src->sp_dn.s3.c1;
   }
   else if ( colorindex == 2 ){
     dest[0]+= src->sp_up.s0.c2;
     dest[1]+= src->sp_up.s1.c2;
     dest[2]+= src->sp_up.s2.c2;
     dest[3]+= src->sp_up.s3.c2;
     dest[4]+= src->sp_dn.s0.c2;
     dest[5]+= src->sp_dn.s1.c2;
     dest[6]+= src->sp_dn.s2.c2;
     dest[7]+= src->sp_dn.s3.c2;
   }
}
static void trace_in_space(_Complex double *dest, _Complex double *source, int idx){
     int i;
     for (i=0; i<8;++i){
       dest[g_coord[idx][TUP]*8+i]+= source[i];
     }
}
static void trace_in_flavor(_Complex double *dest, _Complex double *source, int f1){
     int i;
     for (i=0; i<T_global; ++i){
        dest[i]+= source[2*i+f1];
     }
}

void density_density_1234_s0s0( bispinor ** propfields, int type_1234 ){
   int ix,i;
   int f1,c1,s1;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   int type;
   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( ( type_1234 == TYPE_1 )|| ( type_1234 == TYPE_3 ) ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( ( type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     fprintf(stdout,"Wrong argument for type_ab, it can only be TYPE_1, TYPE_2, TYPE_3 or TYPE_4\n");
     exit(1);
   }

   for (i=0; i<T_global; ++i)
      flavortrace[i]=0.;
//Trace over flavor space
   for (f1=0; f1<2; ++f1){
//Trace over the spinor indices
      for (i=0; i<2*T_global; ++i)
         spinortrace[i]=0.;

      for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
         for (i=0; i<8*T_global; ++i)
            spacetrace[i]=0.;

         for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites
            for (i=0; i<8; ++i)
               colortrace[i]=0.;
            for (c1=0; c1<3; ++c1){
/*   
       TYPE  1 OR  2            (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  3 OR  4            (1+g5)/2*S(x  ,ytilde) running indices bispinor
*/

//for the up quark
               if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){
                 _vector_null( running.sp_up.s0 );
                 _vector_null( running.sp_up.s1 );
                 _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                 _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
                 _vector_null( running.sp_dn.s0 );
                 _vector_null( running.sp_dn.s1 );
                 _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                 _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
               }
               else if ((type_1234 == TYPE_3) || ( type_1234 == TYPE_4) ){
                 _vector_null( running.sp_up.s2 );
                 _vector_null( running.sp_up.s3 );
                 _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
                 _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
                 _vector_null( running.sp_dn.s2 );
                 _vector_null( running.sp_dn.s3 );
                 _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
                 _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );
               }

/*   
       TYPE  1 OR  2     phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
               if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2)){
                 taui_scalarfield_spinor_s0s0( &running, &running, GAMMA_DN, ix, NODIR, DAGGER );
               }
               else if ( (type_1234 == TYPE_3) || (type_1234 == TYPE_4) ){
                 taui_scalarfield_spinor_s0s0( &running, &running, GAMMA_UP, ix, NODIR, NO_DAGG);
               }
/*   
       TYPE  1 OR  2     S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
               multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

               trace_in_color(colortrace,&running,c1);

            }  //End of trace color

            trace_in_space(spacetrace,colortrace,ix);

         } //End of trace space
 
//Gather the results from all nodes to complete the trace in space

#if defined MPI
         for (i=0; i<8*T_global; ++i){
            _Complex double tmp;
            MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
            spacetrace[i]= tmp;
         }
#endif
         trace_in_spinor(spinortrace, spacetrace, s1);
      }

//End of trace in spinor space
/*   
       TYPE  1      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  2      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
       TYPE  4      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
      if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_3) ){
         taui_scalarfield_flavoronly_s0s0( spinortrace, NO_DAGG );
      }
      else if ( (type_1234 == TYPE_4) || ( type_1234 == TYPE_2) ){
         taui_scalarfield_flavoronly_s0s0( spinortrace, DAGGER  );
      }

      trace_in_flavor( flavortrace, spinortrace, f1 );
   } //End of traCe in flavor space

   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){printf("Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("DDS0S0 %d %.3d %10.10e %10.10e\n", type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
      }
   }
   free(flavortrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

}


void density_density_1234( bispinor ** propfields, int type_1234 ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;
   int type;

   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( ( type_1234 == TYPE_1 )|| ( type_1234 == TYPE_3 ) ) {
     spinorstart=0; 
     spinorend  =2;
   }
   else if ( ( type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     if (g_cart_id ==0) fprintf(stdout, "Wrong arument for type_1234, it can only be TYPE_1, TYPE_2, TYPE_3, TYPE_4\n");
     exit(1);
  }

//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over up and down flavors
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
         for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;

         for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
            for (i=0; i<8*T_global; ++i)
               spacetrace[i]=0.;

            for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites

               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE  1 OR  2            (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  3 OR  4            (1+g5)/2*S(x  ,ytilde) running indices bispinor
*/

//for the up quark
                  if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){
                    _vector_null( running.sp_up.s0 );
                    _vector_null( running.sp_up.s1 );
                    _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                    _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
                    _vector_null( running.sp_dn.s0 );
                    _vector_null( running.sp_dn.s1 );
                    _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                    _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
                  }
                  else if ((type_1234 == TYPE_3) || ( type_1234 == TYPE_4)){
                    _vector_null( running.sp_up.s2 );
                    _vector_null( running.sp_up.s3 );
                    _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
                    _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
                    _vector_null( running.sp_dn.s2 );
                    _vector_null( running.sp_dn.s3 );
                    _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
                    _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );
                  }

/*   
       TYPE  1 OR  2     phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
                  if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2)){
                    taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, NODIR, DAGGER );
                  }
                  else if ( (type_1234 == TYPE_3) || (type_1234 == TYPE_4) ){
                    taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, NODIR, NO_DAGG);
                  }
/*   
       TYPE  1 OR  2     S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
                  multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace,&running,c1);

               }  //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space(spacetrace,colortrace,ix);

            } //End of trace space

//Gather the results from all nodes to complete the trace in space
#if defined MPI
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
               spacetrace[i]= tmp;
            }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);
         }//End of trace in spinor space
/*   
       TYPE  1      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  2      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
       TYPE  4      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
         if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_3) ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( (type_1234 == TYPE_4) || ( type_1234 == TYPE_2) ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );
      } //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices

   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){printf("Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("DD %d %.3d %10.10e %10.10e\n", type, i, creal(paulitrace[i])/4.,cimag(paulitrace[i])/4.);
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

}
void density_density_1234_sxsx( bispinor ** propfields, int type_1234 ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;
   int type;

   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( ( type_1234 == TYPE_1 )|| ( type_1234 == TYPE_3 ) ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( ( type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     if (g_cart_id ==0) fprintf(stdout, "Wrong arument for type_1234, it can only be TYPE_1, TYPE_2, TYPE_3, TYPE_4\n");
     exit(1);
  }

//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over up and down flavors
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
         for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;

         for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
            for (i=0; i<8*T_global; ++i)
               spacetrace[i]=0.;

            for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites

               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE  1 OR  2            (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  3 OR  4            (1+g5)/2*S(x  ,ytilde) running indices bispinor
*/

//for the up quark
                  if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){
                    _vector_null( running.sp_up.s0 );
                    _vector_null( running.sp_up.s1 );
                    _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                    _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
                    _vector_null( running.sp_dn.s0 );
                    _vector_null( running.sp_dn.s1 );
                    _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                    _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
                  }
                  else if ((type_1234 == TYPE_3) || ( type_1234 == TYPE_4)){
                    _vector_null( running.sp_up.s2 );
                    _vector_null( running.sp_up.s3 );
                    _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
                    _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
                    _vector_null( running.sp_dn.s2 );
                    _vector_null( running.sp_dn.s3 );
                    _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
                    _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );
                  }

/*   
       TYPE  1 OR  2     phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
                  if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2)){
                    taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, NODIR, DAGGER );
                  }
                  else if ( (type_1234 == TYPE_3) || (type_1234 == TYPE_4) ){
                    taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, NODIR, NO_DAGG);
                  }
/*   
       TYPE  1 OR  2     S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
                  multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace,&running,c1);

               }  //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space(spacetrace,colortrace,ix);

            } //End of trace space

//Gather the results from all nodes to complete the trace in space
#if defined MPI
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
               spacetrace[i]= tmp;
            }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);
         }//End of trace in spinor space
/*   
       TYPE  1      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  2      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
       TYPE  4      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
         if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_3) ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( (type_1234 == TYPE_4) || ( type_1234 == TYPE_2) ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );
      } //End of trace in flavor space
      type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
      if (g_cart_id == 0){printf("Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
        for (i=0; i<T_global; ++i){
          if (g_cart_id == 0){
            printf("DDS%dS%d %d %.3d %10.10e %10.10e\n", tauindex+1, tauindex+1, type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
          }
        }
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices

   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){printf("Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("DD %d %.3d %10.10e %10.10e\n", type, i, creal(paulitrace[i])/4.,cimag(paulitrace[i])/4.);
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}




void naivedirac_current_density_12ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   su3 * restrict upm;
   bispinor running;
#if defined MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( type_ab == TYPE_A ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
    if (g_cart_id == 0){
      fprintf(stdout, "Wrong argument for type_ab, it can only be TYPE_A or TYPE_B\n");exit(1);}                                                                                                                                                                  
   }


//Doing the neccessary communication
#if defined MPI
   for (s1=spinorstart; s1<spinorend; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses); 
            count=0; 
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN, request, &count);
            MPI_Waitall( count, request, statuses);
         }
   free(request);
#endif
   
//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

   for (tauindex=0; tauindex<3; ++tauindex){
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over the spinor indices
         for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;

         for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
            for (i=0; i<8*T_global; ++i)
               spacetrace[i]=0.;
            for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites

               for (i=0; i<8; ++i) 
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){    
/*   
       TYPE  IA OR  IB     U0(x-0)*       (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE IIA OR IIB     U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde) running indices bispinor
*/
                  upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

//for the up quark
                  _vector_null( running.sp_up.s0 ); 
                  _vector_null( running.sp_up.s1 ); 
             
                  if  ( type_12 == TYPE_I ){
                    _su3_multiply( running.sp_up.s2, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 ); 
                    _su3_multiply( running.sp_up.s3, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 ); 


                    _complex_times_vector(running.sp_up.s2,phase_0,running.sp_up.s2);
                    _complex_times_vector(running.sp_up.s3,phase_0,running.sp_up.s3);
                  }
                  else if ( type_12 == TYPE_II ){
                    _su3_inverse_multiply( running.sp_up.s2, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_up.s2 ); 
                    _su3_inverse_multiply( running.sp_up.s3, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_up.s3 ); 

                    _complexcjg_times_vector(running.sp_up.s2,phase_0,running.sp_up.s2);
                    _complexcjg_times_vector(running.sp_up.s3,phase_0,running.sp_up.s3);

                  }


//for the dn quark
                  _vector_null( running.sp_dn.s0 ); 
                  _vector_null( running.sp_dn.s1 ); 
                  if  ( type_12 == TYPE_I ){
                    _su3_multiply( running.sp_dn.s2, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 ); 
                    _su3_multiply( running.sp_dn.s3, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );

                    _complex_times_vector(running.sp_dn.s2,phase_0,running.sp_dn.s2);
                    _complex_times_vector(running.sp_dn.s3,phase_0,running.sp_dn.s3);
 
                  }
                  else if ( type_12 == TYPE_II ){
                    _su3_inverse_multiply( running.sp_dn.s2, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_dn.s2 );
                    _su3_inverse_multiply( running.sp_dn.s3, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_dn.s3 );

                    _complexcjg_times_vector(running.sp_dn.s2,phase_0,running.sp_dn.s2);
                    _complexcjg_times_vector(running.sp_dn.s3,phase_0,running.sp_dn.s3);

                  }



/*   
       TYPE  IA OR  IB     gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/
                  _vector_add_assign(running.sp_up.s0, running.sp_up.s2);
                  _vector_add_assign(running.sp_up.s1, running.sp_up.s3);
                  _vector_null(running.sp_up.s2);
                  _vector_null(running.sp_up.s3);

                  _vector_add_assign(running.sp_dn.s0, running.sp_dn.s2);
                  _vector_add_assign(running.sp_dn.s1, running.sp_dn.s3);
                  _vector_null(running.sp_dn.s2);
                  _vector_null(running.sp_dn.s3);

/*   
       TYPE  IA OR  IB     tau_i*gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/
                  taui_spinor( &running, &running, tauindex  );

/*   
       TYPE  IA OR  IB     S(ytilde, x-0)* tau_i*gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     S(ytilde, x  )* tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/ 
                  if ( type_12 == TYPE_I ){ 
                    multiply_backward_propagator(&running, propfields, &running, ix, TDOWN);
                  }
                  else if ( type_12 == TYPE_II ){
                    multiply_backward_propagator(&running, propfields, &running, ix,NODIR);
                  }
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace,&running,c1);

               }  //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space(spacetrace,colortrace,ix);

            } //End of trace space

//Gather the results from all nodes to complete the trace in space
#if defined MPI
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
               spacetrace[i]= tmp;
            }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor components
            trace_in_spinor(spinortrace, spacetrace, s1);

         }//End of trace in spinor space
   
/*   
       TYPE  IA tau_i*phi(ytilde)        *  (1+gamma5)/2  *   S(ytilde, x-0)*   tau_i*gamma0*U0(x-0)*       (1-g5)/2*   S(x  ,ytilde)
       TYPE  IB phi ^dagger(ytilde)*tau_i *  (1-gamma5)/2  *   S(ytilde, x-0)*   tau_i*gamma0*U0(x-0)*       (1-g5)/2*   S(x  ,ytilde)

       TYPE IIA tau_i*phi(ytilde)        *  (1+gamma5)/2  *   S(ytilde, x  )*   tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*   S(x-0,ytilde)
       TYPE IIB phi^dagger(ytilde)*tau_i *  (1-gamma5)/2  *   S(ytilde, x  )*   tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*   S(x-0,ytilde)

*/
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices 
         trace_in_flavor( flavortrace, spinortrace, f1 );
      } //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices
   
   if (g_cart_id == 0){printf("NaiveDirac Current Density correlator type (%s %s) results\n", type_12 == TYPE_I ? "I" : "II",type_ab == TYPE_A ? "a" :"b");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("DCD %d %d %.3d %10.10e %10.10e\n",type_12, type_ab,  i, creal(paulitrace[i])/4.,cimag(paulitrace[i])/4.0);
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace); 

}
void wilsonterm_current_density_312ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   su3 * restrict upm;
   su3_vector tmpvec;
   bispinor running;
#if defined MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( type_ab == TYPE_A ) {
      spinorstart=0;
      spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
      spinorstart=2;
      spinorend  =4;
   }
   else{
      if (g_cart_id == 0){fprintf(stdout,"Wrong argument for type_1234, it can only be TYPE_1, TYPE_2,  TYPE_3 or TYPE_4 \n"); exit(1);}
   }


// Doing the neccessary communication
#if defined MPI
   for (s1=spinorstart; s1<spinorend; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TUP   , request, &count );
            MPI_Waitall( count, request, statuses); 
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN , request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN , request, &count );
            MPI_Waitall( count, request, statuses);
         }
   free(request);
#endif
//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i){
      paulitrace[i]=0.;
   }
   for (tauindex= 0; tauindex <3; ++tauindex){
//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<2*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE III.1.a OR  III.1.b     U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b                    (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  _vector_null(running.sp_up.s2);
                  _vector_null(running.sp_up.s3);
                  _vector_null(running.sp_dn.s2);
                  _vector_null(running.sp_dn.s3);

                  if ( type_12 == TYPE_1){
                    upm = &g_gauge_field[ix][TUP];

//for the up quark
                    _su3_multiply(running.sp_up.s0, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_up.s0);
                    _su3_multiply(running.sp_up.s1, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_up.s1);

//for the down quark
                    _su3_multiply(running.sp_dn.s0, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_dn.s0);
                    _su3_multiply(running.sp_dn.s1, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_dn.s1);


                    upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

//for the up quark

                    _su3_multiply(tmpvec, (*upm), running.sp_up.s0);
                    _vector_assign(  running.sp_up.s0, tmpvec);
                    _su3_multiply(tmpvec, (*upm), running.sp_up.s1);
                    _vector_assign(  running.sp_up.s1, tmpvec);

                    _complex_times_vector(running.sp_up.s0,phase_00,running.sp_up.s0);
                    _complex_times_vector(running.sp_up.s1,phase_00,running.sp_up.s1);




//for the down quark

                    _su3_multiply(tmpvec, (*upm), running.sp_dn.s0);
                    _vector_assign(  running.sp_dn.s0, tmpvec);
                    _su3_multiply(tmpvec, (*upm), running.sp_dn.s1);
                    _vector_assign(  running.sp_dn.s1, tmpvec);

                    _complex_times_vector(running.sp_dn.s0,phase_00,running.sp_dn.s0);
                    _complex_times_vector(running.sp_dn.s1,phase_00,running.sp_dn.s1);



                  }
                  else if ( type_12 == TYPE_2){
                    _vector_assign( running.sp_up.s0, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s0 );
                    _vector_assign( running.sp_up.s1, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s1 );
                    _vector_assign( running.sp_dn.s0, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s0 );
                    _vector_assign( running.sp_dn.s1, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s1 );
                  }
/*   
       TYPE III.1.a OR  III.1.b     tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b     tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, NODIR, NO_DAGG);


/*   
       TYPE III.1.a OR  III.1.b     S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b     S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  multiply_backward_propagator(&running, propfields, &running, ix, TDOWN);

                  //delta( color component of bispinor running, c1) for all spinor and flavor indices                  
                  trace_in_color(colortrace, &running, c1 );
               } //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space( spacetrace, colortrace, ix);
            }  //End of trace in space

//Gather the results from all nodes to complete the trace in space
#if defined MPI
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
               spacetrace[i]= tmp;
            }
#endif

            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices 
            trace_in_spinor(spinortrace, spacetrace, s1);

         } //End of trace in spinor space
         


/*   
       TYPE III.1.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.1.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)

       TYPE III.2.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
       TYPE III.2.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)

*/ 
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );

      } //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices

 
   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeIII results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("WCDPR1 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}
void wilsonterm_current_density_412ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor **propsecneighbour=NULL;
   bispinor **tmpbisp2d=NULL;
   su3 * restrict upm;
   bispinor running;
   su3_vector tmpvec;
#if defined MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
#endif
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

#if defined MPI
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif
   if ( type_ab == TYPE_A ) {
        spinorstart=0;
         spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
        spinorstart=2;
        spinorend  =4;
   }
   else{
       if (g_cart_id == 0) {fprintf(stdout,"Wrong argument for type_1234, it can only be TYPE_1, TYPE_2,  TYPE_3 or TYPE_4 \n");exit(1);}
   }


   if (type_12 == TYPE_2){
/**********************************
Creating U^dagger(x-0)*U^dagger(x-2*0)*S(x-2*0,ytilde) in three steps:
1; Creating U^dagger(x)*S(x,ytilde)
2;  Creating U^dagger(x+0)Ü0^dagger(x)*S(x, ytilde)
3; Gathering two times in direction TDOWN
***********************************/
//      if (g_cart_id == 0) printf("Hier beginnt etwas\n");
      tmpbisp2d= (bispinor **)malloc(sizeof(bispinor *)*24);
      if (tmpbisp2d == NULL) {
         if (g_cart_id == 0) printf("Error in allocating first\n");
         exit(1);
      }
      propsecneighbour=(bispinor **)malloc(sizeof(bispinor *)*24);
      if (propsecneighbour == NULL) {
         if (g_cart_id == 0) printf("Error in allocating second\n");
         exit(1);
      }
      for (i=0; i<24; ++i){
        propsecneighbour[i]=(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
        if (propsecneighbour[i] == NULL){
          printf("Error in mem alloc propsecneighbour\n");
          exit(1);
        }
        tmpbisp2d[i]=(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND); 
        if (tmpbisp2d[i] == NULL){
          printf("Error in mem alloc tmpbisp2d\n");
          exit(1);
        }
      }
//      if (g_cart_id == 0) printf("Allocation is done\n");
      for (i=0; i<24; ++i)
        for (ix=0; ix<VOLUME; ++ix){
//          if (g_cart_id == 0) printf("nulla i %d ix %d\n",i,ix);
          _bispinor_null(tmpbisp2d[i][ix]);
        }
//      if (g_cart_id == 0) printf("Null is done\n");

      for (ix = 0; ix< VOLUME; ++ix)
        for (s1=spinorstart;s1<spinorend; ++s1)
          for (c1=0; c1<3; ++c1)
            for (f1=0; f1<2; ++f1){
              //if (g_cart_id == 0 ) printf("ix= %d\n", ix);
              upm = &g_gauge_field[ix][TUP];

              _vector_null(tmpbisp2d[12*f1 + 3*s1 + c1][ix].sp_up.s2);
              _vector_null(tmpbisp2d[12*f1 + 3*s1 + c1][ix].sp_up.s3);
              _su3_inverse_multiply(tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s0, (*upm), propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s0);
              _su3_inverse_multiply(tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s1, (*upm), propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s1);

              _vector_null(tmpbisp2d[12*f1 + 3*s1 + c1][ix].sp_dn.s2);
              _vector_null(tmpbisp2d[12*f1 + 3*s1 + c1][ix].sp_dn.s3);
              _su3_inverse_multiply(tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s0, (*upm), propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s0);
              _su3_inverse_multiply(tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s1, (*upm), propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s1);

               upm = &g_gauge_field[g_iup[ix][TUP]][TUP];
/*               if ( g_cart_id == 0){
                 if (ix == 10 ){
                   printf("guage field neig %e\n", creal((*upm).c00));
                 }
               }*/
               
               _vector_null( tmpvec );
               _su3_inverse_multiply(tmpvec, (*upm), tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s0);
               _vector_assign(  tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s0, tmpvec);
               _vector_null( tmpvec );
               _su3_inverse_multiply(tmpvec, (*upm), tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s1);
               _vector_assign(  tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s1, tmpvec);

               _complexcjg_times_vector(tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s0,phase_00,tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s0);
               _complexcjg_times_vector(tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s1,phase_00,tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s1);



               _vector_null( tmpvec );
               _su3_inverse_multiply(tmpvec, (*upm), tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s0);
               _vector_assign(  tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s0, tmpvec);
               _vector_null( tmpvec );
               _su3_inverse_multiply(tmpvec, (*upm), tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s1);
               _vector_assign(  tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s1, tmpvec);

               _complexcjg_times_vector(tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s0,phase_00,tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s0);
               _complexcjg_times_vector(tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s1,phase_00,tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s1);


            }
      
#if defined MPI
      for (s1=spinorstart;s1<spinorend; ++s1)
        for (c1=0; c1<3; ++c1)
          for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( tmpbisp2d[12*f1+3*s1+c1], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
          }
#endif
      for (s1=spinorstart;s1<spinorend; ++s1)
        for (c1=0; c1<3; ++c1)
          for (f1=0; f1<2; ++f1){
            for (ix=0; ix<VOLUMEPLUSRAND; ++ix)
               _bispinor_null(propsecneighbour[12*f1+3*s1+c1][ix]);
            for (ix=0; ix<VOLUME; ++ix){
               _spinor_assign( propsecneighbour[12*f1+3*s1+c1][ix].sp_up, tmpbisp2d[12*f1+3*s1+c1][g_idn[ix][TUP]].sp_up);
               _spinor_assign( propsecneighbour[12*f1+3*s1+c1][ix].sp_dn, tmpbisp2d[12*f1+3*s1+c1][g_idn[ix][TUP]].sp_dn);
            }
          }
#if defined MPI
      for (s1=spinorstart;s1<spinorend; ++s1)
       for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
           count=0;
           generic_exchange_direction_nonblocking( propsecneighbour[12*f1+3*s1+c1], sizeof(bispinor), TDOWN, request, &count );
           MPI_Waitall( count, request, statuses);
        }
#endif
   }
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

// Trace over the Pauli matrices
   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<2*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){

/*   
       TYPE IV.1.a OR  IV.1.b                                     (1+gamma5)/2*S(x    ,ytilde)
       TYPE IV.2.a OR  IV.2.b     U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
*/
                  _vector_null(running.sp_up.s2);
                  _vector_null(running.sp_up.s3);
                  _vector_null(running.sp_dn.s2);
                  _vector_null(running.sp_dn.s3);
                  if ( type_12 == TYPE_2){
//for the up quark
                     _vector_assign( running.sp_up.s0, propsecneighbour[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_up.s0);
                     _vector_assign( running.sp_up.s1, propsecneighbour[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_up.s1);

//for the down quark
                     _vector_assign( running.sp_dn.s0, propsecneighbour[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_dn.s0);
                     _vector_assign( running.sp_dn.s1, propsecneighbour[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_dn.s1);
                  }
                  else if ( type_12 == TYPE_1){
                     _vector_assign( running.sp_up.s0, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s0 );
                     _vector_assign( running.sp_up.s1, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s1 );

                     _vector_assign( running.sp_dn.s0, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s0 );
                     _vector_assign( running.sp_dn.s1, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s1 );
                  }
/*   
       TYPE IV.1.a OR  IV.1.b   tau_i*phi(x)*                                  (1+gamma5)/2*S(x+   ,ytilde)
       TYPE IV.2.a OR  IV.2.b   tau_i*phi(x)*  U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
*/
                  taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, TDOWN, NO_DAGG);

/*   
       TYPE IV.1.a OR  IV.1.b   S(ytilde, x)*tau_i*phi(x)*                                  (1+gamma5)/2*S(x+   ,ytilde)
       TYPE IV.2.a OR  IV.2.b   S(ytilde, x)*tau_i*phi(x)*  U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
*/

                  multiply_backward_propagator(&running, propfields, &running, ix, NODIR);

/*   
       TYPE IV.1.a tau_i*phi(ytilde)*         S(ytilde, x)*tau_i*phi(x)*                                  (1+gamma5)/2*S(x+   ,ytilde)
       TYPE IV.1.b phi^dagger(ytilde)*tau_i*  S(ytilde, x)*tau_i*phi(x)*                                  (1+gamma5)/2*S(x+   ,ytilde)

       TYPE IV.2.a tau_i*phi(ytilde)*         S(ytilde, x)*tau_i*phi(x)*  U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
       TYPE IV.2.b phi^dagger(ytilde)*tau_i*  S(ytilde, x)*tau_i*phi(x)*  U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
*/
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace, &running, c1 );
               } //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space( spacetrace, colortrace, ix);
            }  //End of trace in space

//Gather the results from all nodes to complete the trace in space
#if defined MPI
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
               spacetrace[i]= tmp;
            }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);

         } //End of trace in spinor space

         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );

      } //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices


   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeIV results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("WCDPR2 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab,i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace); 

   if (type_12 == TYPE_2){
     for(i=0;i<24;++i){
       free(tmpbisp2d[i]);
       free(propsecneighbour[i]);
     }
     free(tmpbisp2d);
     free(propsecneighbour);
   }
#if defined MPI
   free(request);
#endif 
}
void wilsonterm_current_density_512ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   su3 * restrict upm;
   bispinor running;
   su3_vector tmpvec;
#if defined MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( type_ab == TYPE_A ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     if (g_cart_id == 0) fprintf(stdout,"Wrong argument for type_1234, it can only be TYPE_1, TYPE_2,  TYPE_3 or TYPE_4 \n");                                                                      
     exit(1);                                                                                                                                                                  
   }
#if defined MPI
//Doing the neccesary communication
   for (s1=spinorstart; s1<spinorend; ++s1)
     for (c1=0; c1<3; ++c1)
       for (f1=0; f1<2; ++f1){
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN , request, &count );
           MPI_Waitall( count, request, statuses);
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TUP   , request, &count );
           MPI_Waitall( count, request, statuses);
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN , request, &count);
           MPI_Waitall( count, request, statuses);
       }
   free(request);
#endif
   for (i=0; i<T_global; ++i)
       paulitrace[i]=0.;
// Trace over the Pauli matrices
   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<2*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE V.1.a OR  V.1.b     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b                                  (1-gamma5)/2 *  S(x-0,ytilde)
*/
                  _vector_null(running.sp_up.s0);
                  _vector_null(running.sp_up.s1);
                  _vector_null(running.sp_dn.s0);
                  _vector_null(running.sp_dn.s1);
                  _vector_null(running.sp_up.s2);
                  _vector_null(running.sp_up.s3);
                  _vector_null(running.sp_dn.s2);
                  _vector_null(running.sp_dn.s3);

                  if ( type_12 == TYPE_1){
                    upm = &g_gauge_field[g_idn[ix][TUP]][TUP];
//for the up quark
                    _su3_inverse_multiply(running.sp_up.s2, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s2);
                    _su3_inverse_multiply(running.sp_up.s3, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s3);

//for the down quark
                    _su3_inverse_multiply(running.sp_dn.s2, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s2);
                    _su3_inverse_multiply(running.sp_dn.s3, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s3);

                    upm = &g_gauge_field[ix][TUP];

//for the up quark
                    _vector_null( tmpvec );
                    _su3_inverse_multiply(tmpvec, (*upm), running.sp_up.s2);
                    _vector_assign(  running.sp_up.s2, tmpvec);
 
                    _vector_null( tmpvec );
                    _su3_inverse_multiply(tmpvec, (*upm), running.sp_up.s3);
                    _vector_assign(  running.sp_up.s3, tmpvec);

                    _complexcjg_times_vector(running.sp_up.s2,phase_00,running.sp_up.s2);
                    _complexcjg_times_vector(running.sp_up.s3,phase_00,running.sp_up.s3);


//for the down quark

                    _vector_null( tmpvec );
                    _su3_inverse_multiply(tmpvec, (*upm), running.sp_dn.s2);
                    _vector_assign(  running.sp_dn.s2, tmpvec);

                    _vector_null( tmpvec );
                    _su3_inverse_multiply(tmpvec, (*upm), running.sp_dn.s3);
                    _vector_assign(  running.sp_dn.s3, tmpvec);


                    _complexcjg_times_vector(running.sp_dn.s2,phase_00,running.sp_dn.s2);
                    _complexcjg_times_vector(running.sp_dn.s3,phase_00,running.sp_dn.s3);

                  }
                  else if ( type_12 == TYPE_2){
                    _vector_assign( running.sp_up.s2, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s2 );
                    _vector_assign( running.sp_up.s3, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s3 );

                    _vector_assign( running.sp_dn.s2, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s2 );
                    _vector_assign( running.sp_dn.s3, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s3 );
                  }
/*   
       TYPE V.1.a OR  V.1.b     phi^dagger(x)*tau_i*     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b     phi^dagger(x)*tau_i                                        (1-gamma5)/2 *  S(x-0,ytilde)
*/
                 
                  taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, NODIR, DAGGER);

/*   
       TYPE V.1.a OR  V.1.b     S(ytilde,x+0)*phi^dagger(x)*tau_i*     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b     S(ytilde,x-0)*phi^dagger(x)*tau_i                                   (1-gamma5)/2 *  S(x-0,ytilde)
*/

                  if (type_12 == TYPE_1){
                    multiply_backward_propagator(&running, propfields, &running, ix, TUP);
                  }
                  else if (type_12 == TYPE_2){
                    multiply_backward_propagator(&running, propfields, &running, ix, TDOWN);
                  }
/*   
        TYPE V.1.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE V.1.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)

       TYPE V.2.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)

*/
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace, &running, c1 );
               } //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space( spacetrace, colortrace, ix);
            }  //End of trace in space
#if defined MPI
//Gather the results from all nodes to complete the trace in space
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
               spacetrace[i]= tmp;
            }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);

         } //End of trace in spinor space

         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );

      } //End of trace in flavor space

      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices


   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeV results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("WCDPL1 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}
void wilsonterm_current_density_612ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor **starting2d;
   bispinor running;
   su3 * restrict upm;
   su3_vector tmpvec;
#if defined MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
#endif
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

#if defined MPI
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif
   if ( type_ab == TYPE_A ) {
        spinorstart=0;
        spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
        spinorstart=2;
        spinorend  =4;
   }
   else{
       if (g_cart_id == 0) fprintf(stdout,"Wrong argument for type_1234, it can only be TYPE_1, TYPE_2,  TYPE_3 or TYPE_4 \n");                                                                      
       exit(1);                                                                                                                                                                  
   }
   if (type_12 == TYPE_1){
     for (i=0; i<T_global; ++i)
       paulitrace[i]=0.;
     for (tauindex=0; tauindex<3; ++tauindex){
       for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;
       for (f1=0; f1<2; ++f1){
         for (i=0; i<2*T_global; ++i)
           spinortrace[i]=0.;
         for (s1= spinorstart; s1<spinorend; ++s1){
           for (i=0; i<8*T_global; ++i)
             spacetrace[i]=0.;
           for (ix = 0; ix< VOLUME; ++ix){
             for (i=0; i<8; ++i)
               colortrace[i]=0.;
             for (c1=0; c1<3; ++c1){
               _vector_null( running.sp_up.s0 );
               _vector_null( running.sp_up.s1 );
               _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
               _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
               _vector_null( running.sp_dn.s0 );
               _vector_null( running.sp_dn.s1 );
               _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
               _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
             
               taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, TDOWN, DAGGER );
                 
               multiply_backward_propagator(&running, propfields, &running, ix, NODIR );
  
               trace_in_color(colortrace,&running,c1);
             }  //End of trace color
              //sum over all lattice sites the result of the color trace
             trace_in_space(spacetrace,colortrace,ix);
           } //End of trace space
#if defined MPI
           for (i=0; i<8*T_global; ++i){
             _Complex double tmp;
             MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
             spacetrace[i]= tmp;
           }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
           trace_in_spinor(spinortrace, spacetrace, s1);
         }//End of trace in spinor space
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         trace_in_flavor( flavortrace, spinortrace, f1 );
       } //End of trace in flavor space
      //sum for all Pauli matrices
       for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
     } //End of trace for Pauli matrices
     if  (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeVI results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
     for (i=0; i<T_global; ++i){
       if (g_cart_id == 0){
        printf("WCDPL2 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
       }
     }
   }
   if (type_12 == TYPE_2 ){
      starting2d=(bispinor **)malloc(sizeof(bispinor *)*3);
      for (i=0; i<3; ++i){
        starting2d[i] =(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
        if (starting2d[i] == NULL){
          if (g_cart_id == 0) printf("Memory allocation error starting2d VI\n");
          exit(1);
        }
        for (ix=0; ix<VOLUME; ++ix){
          _bispinor_null(starting2d[i][ix]);
        }
      } 
//Doing the neccesary communication
#if defined MPI
      for (s1=spinorstart; s1<spinorend; ++s1)
        for (c1=0; c1<3; ++c1)
           for (f1=0; f1<2; ++f1){
             count=0;
             generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TUP   , request, &count );
             MPI_Waitall( count, request, statuses);
             count=0;
             generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN   , request, &count );
             MPI_Waitall( count, request, statuses);
          }
#endif
        for (i=0; i<T_global; ++i)
          paulitrace[i]=0.;
        for (tauindex=0; tauindex<3; ++tauindex){
          for (i=0; i<T_global; ++i)
            flavortrace[i]=0.;
          for (f1=0; f1<2; ++f1){
            for (i=0; i<2*T_global; ++i)
              spinortrace[i]=0.;
            for (s1= spinorstart; s1<spinorend; ++s1){
              for (i=0; i<8*T_global; ++i)
                spacetrace[i]=0.;
              for (ix = 0; ix< VOLUME; ++ix){
                for (i=0; i<8; ++i)
                  colortrace[i]=0.;
                for (c1=0; c1<3; ++c1){
                  _vector_null( starting2d[c1][ix].sp_up.s0 );
                  _vector_null( starting2d[c1][ix].sp_up.s1 );

                  _vector_null( starting2d[c1][ix].sp_dn.s0 );
                  _vector_null( starting2d[c1][ix].sp_dn.s1 );

                  upm = &g_gauge_field[ix][TUP];

                  _su3_multiply(starting2d[c1][ix].sp_up.s2, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_up.s2);
                  _su3_multiply(starting2d[c1][ix].sp_up.s3, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_up.s3);

                  _su3_multiply(starting2d[c1][ix].sp_dn.s2, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_dn.s2);
                  _su3_multiply(starting2d[c1][ix].sp_dn.s3, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_dn.s3);

                  upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

                  _su3_multiply(tmpvec, (*upm), starting2d[c1][ix].sp_up.s2);
                  _vector_assign(  starting2d[c1][ix].sp_up.s2, tmpvec);
                  _su3_multiply(tmpvec, (*upm), starting2d[c1][ix].sp_up.s3);
                  _vector_assign(  starting2d[c1][ix].sp_up.s3, tmpvec);

                  _su3_multiply(tmpvec, (*upm), starting2d[c1][ix].sp_dn.s2);
                  _vector_assign(  starting2d[c1][ix].sp_dn.s2, tmpvec);
                  _su3_multiply(tmpvec, (*upm), starting2d[c1][ix].sp_dn.s3);
                  _vector_assign(  starting2d[c1][ix].sp_dn.s3, tmpvec);

                  _complex_times_vector(starting2d[c1][ix].sp_up.s2,phase_00,starting2d[c1][ix].sp_up.s2);
                  _complex_times_vector(starting2d[c1][ix].sp_up.s3,phase_00,starting2d[c1][ix].sp_up.s3);
                  _complex_times_vector(starting2d[c1][ix].sp_dn.s2,phase_00,starting2d[c1][ix].sp_dn.s2);
                  _complex_times_vector(starting2d[c1][ix].sp_dn.s3,phase_00,starting2d[c1][ix].sp_dn.s3);
             
                  taui_scalarfield_spinor( &starting2d[c1][ix], &starting2d[c1][ix], GAMMA_DN, tauindex, ix, NODIR, DAGGER );

                  multiply_backward_propagator(&starting2d[c1][ix], propfields, &starting2d[c1][ix], ix, TDOWN);

                }
              }
              for (c1=0; c1<3; ++c1){
                count=0;
                generic_exchange_direction_nonblocking( starting2d[c1], sizeof(bispinor), TDOWN, request, &count );
                MPI_Waitall( count, request, statuses);
              }
              for (ix=0; ix<VOLUME; ++ix){
                for (i=0; i<8; ++i)
                  colortrace[i]=0.;
                for (c1=0; c1<3; ++c1)
                  trace_in_color(colortrace,&starting2d[c1][g_idn[ix][TUP]],c1);
                trace_in_space(spacetrace,colortrace,ix);
              } 
#if defined MPI
              for (i=0; i<8*T_global; ++i){
                _Complex double tmp;
                MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
                spacetrace[i]= tmp;
              }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
              trace_in_spinor(spinortrace, spacetrace, s1);
            }//End of trace in spinor space
            if ( type_ab == TYPE_A ){
              taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
            }
            else if ( type_ab == TYPE_B ){
              taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
            }
            trace_in_flavor( flavortrace, spinortrace, f1 );
          } //End of trace in flavor space
      //sum for all Pauli matrices
          for (i=0;i<T_global; ++i)
            paulitrace[i]+=flavortrace[i];
        } //End of trace for Pauli matrices
        if  (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeVI results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
        for (i=0; i<T_global; ++i){
          if (g_cart_id == 0){
            printf("WCDPL2 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
          }
        }
      for (i=0; i<3; ++i){
        free(starting2d[i]);
      } 
      free(starting2d);
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace); 
#if defined MPI
   if (type_12 == TYPE_2)
     free(request);
#endif
}
