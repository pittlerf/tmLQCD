/***********************************************************************
 * Copyright (C) 2015 Bartosz Kostrzewa
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
 *  
 * File: mixed_cg_her.c
 *
 * CG solver for hermitian f only!
 *
 * The externally accessible functions are
 *
 *
 *   int cg(spinor * const P, spinor * const Q, double m, const int subtract_ev)
 *     CG solver
 *
 * input:
 *   m: Mass to be use in D_psi
 *   subtrac_ev: if set to 1, the lowest eigenvectors of Q^2 will
 *               be projected out.
 *   Q: source
 * inout:
 *   P: initial guess and result
 * 
 *
 **************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "su3.h"
#include "linalg_eo.h"
#include "start.h"
#include "operator/tm_operators_32.h"
#include "solver/matrix_mult_typedef.h"
#include "read_input.h"

#include "solver_field.h"
#include "solver/mixed_cg_her.h"
#include "gettime.h"

#define DELTA 1.0e-4

static inline unsigned int inner_loop(spinor32 * const x, spinor32 * const p, spinor32 * const q, spinor32 * const r, float * const rho1, float delta,
                              matrix_mult32 f32, const float eps_sq, const unsigned int max_inner_it, const unsigned int N, const unsigned int iter ){

  static float alpha, beta, rho;
  static unsigned int j;

  rho = *rho1;
  delta = delta*rho;

  for(j = 0; j < max_inner_it; ++j){

    f32(q,p);
    alpha = rho/scalar_prod_r_32(p,q,N,1);
    assign_add_mul_r_32(x, p, alpha, N);
    assign_add_mul_r_32(r, q, -alpha, N);
    rho = square_norm_32(r,N,1);
    beta = rho / *rho1;
    *rho1 = rho;
    assign_mul_add_r_32(p, beta, r, N);

    if( rho < delta || rho < eps_sq ) break;
    
    if(g_debug_level > 2 && g_proc_id == 0) {
      printf("inner CG: %d res^2 %g\t\n", j+iter, rho);
    }
  }

  return j;
}


/* P output = solution , Q input = source */
int mixed_cg_her(spinor * const P, spinor * const Q, const int max_iter, 
		 double eps_sq, const int rel_prec, const int N, matrix_mult f, matrix_mult32 f32) {

  int i = 0, iter = 0, j = 0;
  float sqnrm = 0., sqnrm2, squarenorm;
  float pro, err, alpha_cg, beta_cg, rho;
  double sourcesquarenorm, sqnrm_d, squarenorm_d;
  spinor *delta, *y, *xhigh;
  spinor32 *x, *stmp, *p, *q, *r;
  spinor ** solver_field = NULL;
  spinor32 ** solver_field32 = NULL;  
  const int nr_sf = 3;
  const int nr_sf32 = 4;

  double shigh, shighp1;
  spinor *rhigh, *qhigh;

  int max_inner_it = mixcg_maxinnersolverit;
  int N_outer = max_iter/max_inner_it;
  //to be on the save side we allow at least 10 outer iterations
  if(N_outer < 10) N_outer = 10;
  
  int save_sloppy = g_sloppy_precision_flag;
  double atime, etime, flops;
  
  if(N == VOLUME) {
    init_solver_field(&solver_field, VOLUMEPLUSRAND, nr_sf);    
    init_solver_field_32(&solver_field32, VOLUMEPLUSRAND, nr_sf32);
  }
  else {
    init_solver_field(&solver_field, VOLUMEPLUSRAND/2, nr_sf);
    init_solver_field_32(&solver_field32, VOLUMEPLUSRAND/2, nr_sf32);    
  }

  atime = gettime();

  xhigh = solver_field[2];
  rhigh = solver_field[1];
  qhigh = solver_field[0];

  x = solver_field32[3];
  r = solver_field32[2];
  p = solver_field32[1];
  q = solver_field32[0];

  g_sloppy_precision_flag = 0;

  zero_spinor_field(xhigh,N);
  zero_spinor_field_32(x,N);
  
  assign_to_32(r,Q,N);
  assign_to_32(p,Q,N);
  rho = square_norm_32(r,N,1);
  
  iter += inner_loop(x, p, q, r, &rho, DELTA, f32, (float)eps_sq, max_inner_it, N, iter);

  for(i = 0; i < N_outer; i++) {
     
    ++iter; 
    addto_32(xhigh,x,N);
    f(qhigh,xhigh);
    diff(rhigh,Q,qhigh,N);
    shigh = square_norm(rhigh,N,1);
    
    if(g_debug_level > 2 && g_proc_id == 0) {
      printf("mixed CG: last inner residue: %g\t\n", rho);
      printf("mixed CG: true residue %d %g\t\n", iter, shigh); fflush(stdout);
    }
    if(((shigh <= eps_sq) && (rel_prec == 0)) || ((shigh <= eps_sq*sourcesquarenorm) && (rel_prec == 1))) {
      assign(P,xhigh,N);
      g_sloppy_precision_flag = save_sloppy;
      etime = gettime();     

      if(g_debug_level > 0 && g_proc_id == 0) {
      	if(N != VOLUME){
      	  /* 2 A + 2 Nc Ns + N_Count ( 2 A + 10 Nc Ns ) */
      	  /* 2*1608.0 because the linalg is over VOLUME/2 */
      	  flops = (2*(2*1608.0+2*3*4) + 2*3*4 + iter*(2.*(2*1608.0+2*3*4) + 10*3*4))*N/1.0e6f;
      	  printf("# mixed CG: iter: %d eps_sq: %1.4e t/s: %1.4e\n", iter, eps_sq, etime-atime); 
      	  printf("# mixed CG: flopcount (for e/o tmWilson only): t/s: %1.4e mflops_local: %.1f mflops: %.1f\n", 
      	      etime-atime, flops/(etime-atime), g_nproc*flops/(etime-atime));
      	}
      	else{
      	  /* 2 A + 2 Nc Ns + N_Count ( 2 A + 10 Nc Ns ) */
      	  flops = (2*(1608.0+2*3*4) + 2*3*4 + iter*(2.*(1608.0+2*3*4) + 10*3*4))*N/1.0e6f;      
      	  printf("# mixed CG: iter: %d eps_sq: %1.4e t/s: %1.4e\n", iter, eps_sq, etime-atime); 
      	  printf("# mixed CG: flopcount (for non-e/o tmWilson only): t/s: %1.4e mflops_local: %.1f mflops: %.1f\n", 
      	      etime-atime, flops/(etime-atime), g_nproc*flops/(etime-atime));      
      	}
      }      
      
      g_sloppy_precision_flag = save_sloppy;
      finalize_solver(solver_field, nr_sf);
      finalize_solver_32(solver_field32, nr_sf32); 
      return(iter+i);
    }

    // correct defect
    assign_to_32(r,rhigh,N);
    rho = square_norm_32(r,N,1);

    // project search direction to be orthogonal to corrected residual
    float gamma = scalar_prod_r_32(r,p,N,1);
    assign_add_mul_r_32(p,r,-gamma,N);
    zero_spinor_field_32(x,N);

    iter += inner_loop(x, p, q, r, &rho, DELTA, f32, (float)eps_sq, max_inner_it, N, iter);
    
  }
  g_sloppy_precision_flag = save_sloppy;
  finalize_solver(solver_field, nr_sf);
  finalize_solver_32(solver_field32, nr_sf32); 
  return(-1);
}

