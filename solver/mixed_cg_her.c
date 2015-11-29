/***********************************************************************
 * Copyright (C) 2015 Florian Burger, Bartosz Kostrzewa
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
#define BETA_DP 1
#define PR 0

void output_flops(const double seconds, const unsigned int N, const unsigned int iter, const double eps_sq);

static inline unsigned int inner_loop_high(spinor * const x, spinor * const p, spinor * const q, spinor * const r, double * const rho1, double delta,
                              matrix_mult f, const double eps_sq, const unsigned int max_inner_it, const unsigned int N, const unsigned int iter ){

  static double alpha, beta, rho, rhomax;
  static unsigned int j;

  rho = *rho1;
  rhomax = *rho1;

  for(j = 0; j < max_inner_it; ++j){

    f(q,p);
    alpha = rho/scalar_prod_r(p,q,N,1);
    assign_add_mul_r(x, p, alpha, N);
    assign_add_mul_r(r, q, -alpha, N);
    rho = square_norm(r,N,1);
    beta = rho / *rho1;
    *rho1 = rho;
    assign_mul_add_r(p, beta, r, N);
    
    /* break out of inner loop if iterated residual goes below some fraction of the maximum observed
     * iterated residual since the last update or if the target precision has been reached */
    if( rho < delta*rhomax || rho < eps_sq ) break;
    if( rho > rhomax ) rhomax = rho;
    
    if(g_debug_level > 2 && g_proc_id == 0) {
      printf("DP_inner CG: %d res^2 %g\t\n", j+iter, rho);
    }
  }

  return j;
}

static inline unsigned int inner_loop(spinor32 * const x, spinor32 * const p, spinor32 * const q, spinor32 * const r, float * const rho1, float delta,
                              matrix_mult32 f32, const float eps_sq, const unsigned int max_inner_it, const unsigned int N, const unsigned int iter,
                              float alpha, float beta, int pipelined, int pr ){

  static float rho, rhomax, pro;
  static unsigned int j;

  rho = *rho1;
  rhomax = *rho1;

  if(pipelined==0){
    for(j = 0; j < max_inner_it; ++j){
      f32(q,p);
      pro = scalar_prod_r_32(p,q,N,1);
      alpha = rho/pro;
      assign_add_mul_r_32(x, p, alpha, N);
      assign_add_mul_r_32(r, q, -alpha, N);
      rho = square_norm_32(r,N,1);
      // Polak-Ribiere seems to stabilize the solver, resulting in fewer iterations 
      if(pr){
        beta = alpha*(alpha*square_norm_32(q,N,1)-pro) / *rho1;
      }else{
        beta = rho / *rho1;
      }
      *rho1 = rho;
      assign_mul_add_r_32(p, beta, r, N);
      if(g_debug_level > 2 && g_proc_id == 0) {
        printf("SP_inner CG: %d res^2 %g\t\n", j+iter, rho);
      }
      /* break out of inner loop if iterated residual goes below some fraction of the maximum observed
       * iterated residual since the last update or if the target precision has been reached 
       * enforce convergence more strictly by a factor of 1.3 to avoid unnecessary restarts 
       * if the real residual is still a bit too large */
      if( rho < delta*rhomax || 1.3*rho < eps_sq ) break;
      if( rho > rhomax ) rhomax = rho;
    }
  }else{
    for(j = 0; j < max_inner_it; ++j){
      assign_add_mul_r_32(x, p, alpha, N);
      assign_add_mul_r_32(r, q, -alpha, N);
      assign_mul_add_r_32(p, beta, r, N);
      f32(q,p);
  
      rho = square_norm_32(r,N,1);
      *rho1 = rho;
      pro = scalar_prod_r_32(p,q,N,1);
      alpha = rho/pro;
      beta = alpha*(alpha*square_norm_32(q,N,1)-pro)/rho;
      /* break out of inner loop if iterated residual goes below some fraction of the maximum observed
       * iterated residual since the last update or if the target precision has been reached */
      if(g_debug_level > 2 && g_proc_id == 0) {
        printf("SP_inner CG: %d res^2 %g\t\n", j+iter, rho);
      }
      if( rho < delta*rhomax || 1.3*rho < eps_sq ) break;
      if( rho > rhomax ) rhomax = rho;
    }
  }

  return j;
}


/* P output = solution , Q input = source */
int mixed_cg_her(spinor * const P, spinor * const Q, const int max_iter, 
		 double eps_sq, const int rel_prec, const int N, matrix_mult f, matrix_mult32 f32) {

  int i = 0, iter = 0, j = 0;
  float rho_sp, beta_sp;
  double beta_dp, rho_dp;
  double sourcesquarenorm;

  spinor *xhigh, *rhigh, *qhigh, *phigh;
  spinor32 *x, *p, *q, *r;
  spinor ** solver_field = NULL;
  spinor32 ** solver_field32 = NULL;  
  const int nr_sf = 4;
  const int nr_sf32 = 4;

  float delta = DELTA;

  int max_inner_it = mixcg_maxinnersolverit;
  //int N_outer = max_iter/max_inner_it;
  //to be on the safe side we allow at least 40 outer iterations
  //if(N_outer < 40) N_outer = 40;
  int N_outer = 100;
  
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

  phigh = solver_field[3];
  xhigh = solver_field[2];
  rhigh = solver_field[1];
  qhigh = solver_field[0];

  x = solver_field32[3];
  r = solver_field32[2];
  p = solver_field32[1];
  q = solver_field32[0];

  g_sloppy_precision_flag = 0;

  // should compute real residual here, for now we always use a zero guess
  zero_spinor_field_32(x,N);
  zero_spinor_field(xhigh,N);
  assign(phigh,Q,N);
  assign(rhigh,Q,N);
  
  rho_dp = square_norm(rhigh,N,1);
  assign_to_32(r,rhigh,N);
  rho_sp = square_norm_32(r,N,1);
  assign_32(p,r,N);
  
  iter += inner_loop(x, p, q, r, &rho_sp, delta, f32, (float)eps_sq, max_inner_it, N, iter, 0.0, 0.0, 0, PR);

  for(i = 0; i < N_outer; i++) {
     
    ++iter;
    // update high precision solution 
    addto_32(xhigh,x,N);
    // compute real residual
    f(qhigh,xhigh);
    diff(rhigh,Q,qhigh,N);
    beta_dp = 1/rho_dp;
    rho_dp = square_norm(rhigh,N,1);
    beta_dp *= rho_dp; 

    if(g_debug_level > 2 && g_proc_id == 0) {
      printf("mixed CG last inner residue:       %17g\n", rho_sp);
      printf("mixed CG true residue:             %6d %10g\n", iter, rho_dp); fflush(stdout);
      printf("mixed CG residue reduction factor: %6d %10g\n", iter, beta_dp);
    }
    if(((rho_dp <= eps_sq) && (rel_prec == 0))) { //|| ((shigh <= eps_sq*sourcesquarenorm) && (rel_prec == 1))) {
      // output solution
      assign(P,xhigh,N);
      
      etime = gettime();
      output_flops(etime-atime, N, iter, eps_sq);
      
      g_sloppy_precision_flag = save_sloppy;
      finalize_solver(solver_field, nr_sf);
      finalize_solver_32(solver_field32, nr_sf32); 
      return(iter+i);
    }

    // correct defect
    assign_to_32(r,rhigh,N);
    rho_sp = rho_dp; // not sure if it's fine to truncate this or whether one should calculate it in SP directly

    // throw away search vector if it seems that we're stuck
    if(beta_dp>=5) {
      assign_32(p,r,N);
    }else{
      // otherwise project search vector to be
      // orthogonal to new residual in double precision
      assign_to_64(phigh,p,N);
      assign_mul_add_r(phigh,beta_dp,rhigh,N);
      assign_to_32(p,phigh,N);
    }

    zero_spinor_field_32(x,N);

    iter += inner_loop(x, p, q, r, &rho_sp, delta, f32, (float)eps_sq, max_inner_it, N, iter, 0.0, 0.0, 0, PR);
  }
  g_sloppy_precision_flag = save_sloppy;
  finalize_solver(solver_field, nr_sf);
  finalize_solver_32(solver_field32, nr_sf32); 
  return(-1);
}

void output_flops(const double seconds, const unsigned int N, const unsigned int iter, const double eps_sq){
  double flops;
  if(g_debug_level > 0 && g_proc_id == 0) {
  	if(N != VOLUME){
  	  /* 2 A + 2 Nc Ns + N_Count ( 2 A + 10 Nc Ns ) */
  	  /* 2*1608.0 because the linalg is over VOLUME/2 */
  	  flops = (2*(2*1608.0+2*3*4) + 2*3*4 + iter*(2.*(2*1608.0+2*3*4) + 10*3*4))*N/1.0e6f;
  	  printf("# mixed CG: iter: %d eps_sq: %1.4e t/s: %1.4e\n", iter, eps_sq, seconds); 
  	  printf("# mixed CG: flopcount (for e/o tmWilson only): t/s: %1.4e mflops_local: %.1f mflops: %.1f\n", 
  	      seconds, flops/(seconds), g_nproc*flops/(seconds));
  	}
  	else{
  	  /* 2 A + 2 Nc Ns + N_Count ( 2 A + 10 Nc Ns ) */
  	  flops = (2*(1608.0+2*3*4) + 2*3*4 + iter*(2.*(1608.0+2*3*4) + 10*3*4))*N/1.0e6f;      
  	  printf("# mixed CG: iter: %d eps_sq: %1.4e t/s: %1.4e\n", iter, eps_sq, seconds); 
  	  printf("# mixed CG: flopcount (for non-e/o tmWilson only): t/s: %1.4e mflops_local: %.1f mflops: %.1f\n", 
  	      seconds, flops/(seconds), g_nproc*flops/(seconds));      
  	}
  }      
}
