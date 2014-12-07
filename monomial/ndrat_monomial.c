/***********************************************************************
 *
 * Copyright (C) 2013 Carsten Urbach
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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include "global.h"
#include "su3.h"
#include "su3adj.h"
#include "linalg_eo.h"
#include "start.h"
#include "gettime.h"
#include "solver/solver.h"
#include "deriv_Sb.h"
#include "read_input.h"
#include "init/init_chi_spinor_field.h"
#include "operator/tm_operators.h"
#include "operator/tm_operators_nd.h"
#include "operator/Hopping_Matrix.h"
#include "monomial/monomial.h"
#include "hamiltonian_field.h"
#include "boundary.h"
#include "operator/clovertm_operators.h"
#include "operator/clover_leaf.h"
#include "rational/rational.h"
#include "phmc.h"
#include "ndrat_monomial.h"

#include "dirty_shameful_business.h"
#include "expo.h"
#include "buffers/gauge.h"
#include "buffers/adjoint.h"
#include "measure_gauge_action.h"
#include "update_backward_gauge.h"
#include "operator/clover_leaf.h"

void nd_set_global_parameter(monomial * const mnl) {

  g_mubar = mnl->mubar;
  g_epsbar = mnl->epsbar;
  g_kappa = mnl->kappa;
  g_c_sw = mnl->c_sw;
  boundary(g_kappa);
  phmc_cheb_evmin = mnl->EVMin;
  phmc_invmaxev = mnl->EVMaxInv;
  phmc_cheb_evmax = 1.;
  phmc_Cpol = 1.;
  // used for preconditioning in cloverdetrat
  g_mu3 = 0.;

  return;
}


/********************************************
 *
 * Here \delta S_b is computed
 *
 ********************************************/

void ndrat_derivative(const int id, hamiltonian_field_t * const hf) {
  static short first = 1;
  monomial * mnl = &monomial_list[id];

  if(mnl->write_deriv) {
    char filename[100];
    char mode[2] = {'a','\0'};
    if( first == 1 ) {
      mode[0] = 'w';
      first = 0;
    }
    
    adjoint_field_t df_analytical = get_adjoint_field();
    zero_adjoint_field(&df_analytical);
    ohnohack_remap_df0(df_analytical);
    ndrat_derivative_analytical(id,hf);

    snprintf(filename,100,"%s_%02d_f_analytical.bin",mnl->name,mnl->timescale);
    write_deriv_file(filename, mode, df_analytical, mnl);
    
    if(mnl->num_deriv) {
      adjoint_field_t df_numerical  = get_adjoint_field();
      zero_adjoint_field(&df_numerical);
      ohnohack_remap_df0(df_numerical);
      ndrat_derivative_numerical(id,hf);
      snprintf(filename,100,"%s_%02d_f_numerical.bin",mnl->name,mnl->timescale);
      write_deriv_file(filename, mode, df_numerical, mnl);
      
      int x = 1, mu = 1;
      double *ar_num = (double*)&df_numerical[x][mu];
      double *ar_an = (double*)&df_analytical[x][mu];
      fprintf(stderr, "[DEBUG] Comparison of force calculation at [%d][%d]!\n",x,mu);
      fprintf(stderr, "         numerical force <-> analytical force \n");
      for (int component = 0; component < 8; ++component)
        fprintf(stderr, "    [%d]  %+14.12f <-> %+14.12f\n", component, ar_num[component], ar_an[component]); //*/
      
      return_adjoint_field(&df_numerical);
    }
    
    if(!mnl->decouple) {
      #ifdef OMP
      #pragma omp parallel for
      #endif
      for(int x = 0; x < VOLUME; ++x) {
        for(int mu = 0; mu < 4; ++mu) {
          _add_su3adj(df[x][mu],df_analytical[x][mu]);
        }
      }
    }
    return_adjoint_field(&df_analytical);
    ohnohack_remap_df0(df);
    
  } else { // write_deriv
    if(!mnl->decouple)
      ndrat_derivative_analytical(id,hf);
  }
}

void ndrat_derivative_numerical(const int id, hamiltonian_field_t * const hf) {
  monomial * mnl = &monomial_list[id];
  double atime = gettime();

  su3adj rotation;
  double *ar_rotation = (double*)&rotation;
  double const eps = num_deriv_eps;
  double const oneov2eps = 1.0/(2*eps);
  double const epsilon[2] = {-eps,eps};
  su3 old_value;
  su3 mat_rotation;
  double* xm;
  su3* link;

  for(int x = 0; x < VOLUME; ++x)
  {
    for(int mu = 0; mu < 4; ++mu)
    {
      xm=(double*)&hf->derivative[x][mu];
      for (int component = 0; component < 8; ++component)
      {
        double h_rotated[2] = {0.0,0.0};
        for(int direction = 0; direction < 2; ++direction)
        {
          link=&(hf->gaugefield[x][mu]);
          // save current value of gauge field
          memmove(&old_value, link, sizeof(su3));
          /* Introduce a rotation along one of the components */
          memset(ar_rotation, 0, sizeof(su3adj));
          ar_rotation[component] = epsilon[direction];
          exposu3(&mat_rotation, &rotation);
          _su3_times_su3(*link, mat_rotation, old_value);
          g_update_gauge_copy = 1;
          update_backward_gauge(_AS_GAUGE_FIELD_T(hf->gaugefield));
          
          h_rotated[direction] = ndrat_energy(id,(su3 const **)hf->gaugefield);
          if( mnl->type == NDCLOVERRAT && mnl->trlog ) {
            sw_term( (const su3**) hf->gaugefield, mnl->kappa, mnl->c_sw);
            h_rotated[direction] += -sw_trace_nd(EE, mnl->mubar, mnl->epsbar);
          }

          // reset modified part of gauge field
          memmove(link,&old_value, sizeof(su3));
          g_update_gauge_copy = 1;
          update_backward_gauge(_AS_GAUGE_FIELD_T(hf->gaugefield));
        } // direction
        // calculate force contribution from gauge field due to rotation
        xm[component] += (h_rotated[1]-h_rotated[0])*oneov2eps;
      } // component
    } // mu
  } // x
  double etime = gettime();
  if(g_debug_level > 1 && g_proc_id == 0) {
    printf("# Time for numerical %s monomial derivative: %e s\n", mnl->name, etime-atime);
  }
}

void ndrat_derivative_analytical(const int id, hamiltonian_field_t * const hf) {
  monomial * mnl = &monomial_list[id];
  solver_pm_t solver_pm;
  double atime, etime;
  atime = gettime();
  nd_set_global_parameter(mnl);
  if(mnl->type == NDCLOVERRAT) {
    for(int i = 0; i < VOLUME; i++) { 
      for(int mu = 0; mu < 4; mu++) { 
	_su3_zero(swm[i][mu]);
	_su3_zero(swp[i][mu]);
      }
    }
  
    // we compute the clover term (1 + T_ee(oo)) for all sites x
    sw_term( (const su3**) hf->gaugefield, mnl->kappa, mnl->c_sw); 
    // we invert it for the even sites only
    sw_invert_nd(mnl->mubar*mnl->mubar - mnl->epsbar*mnl->epsbar);
  }
  mnl->forcefactor = mnl->EVMaxInv;

  solver_pm.max_iter = mnl->maxiter;
  solver_pm.squared_solver_prec = mnl->forceprec;
  solver_pm.no_shifts = mnl->rat.np;
  solver_pm.shifts = mnl->rat.mu;
  solver_pm.rel_prec = g_relative_precision_flag;
  solver_pm.type = CGMMSND;
  solver_pm.M_ndpsi = &Qtm_pm_ndpsi;
  if(mnl->type == NDCLOVERRAT) solver_pm.M_ndpsi = &Qsw_pm_ndpsi;
  solver_pm.sdim = VOLUME/2;
  // this generates all X_j,o (odd sites only) -> g_chi_up|dn_spinor_field
  mnl->iter1 += cg_mms_tm_nd(g_chi_up_spinor_field, g_chi_dn_spinor_field,
			     mnl->pf, mnl->pf2,
			     &solver_pm);
  
  for(int j = (mnl->rat.np-1); j > -1; j--) {
    if(mnl->type == NDCLOVERRAT) {
      // multiply with Q_h * tau^1 + i mu_j to get Y_j,o (odd sites)
      // needs phmc_Cpol = 1 to work for ndrat!
      Qsw_tau1_sub_const_ndpsi(mnl->w_fields[0], mnl->w_fields[1],
			       g_chi_up_spinor_field[j], g_chi_dn_spinor_field[j], 
			       -I*mnl->rat.mu[j], 1., mnl->EVMaxInv);
      
      /* Get the even parts X_j,e */
      /* H_eo_... includes tau_1 */
      H_eo_sw_ndpsi(mnl->w_fields[2], mnl->w_fields[3], 
		    g_chi_up_spinor_field[j], g_chi_dn_spinor_field[j]);

    }
    else {
      // multiply with Q_h * tau^1 + i mu_j to get Y_j,o (odd sites)
      // needs phmc_Cpol = 1 to work for ndrat!
      Q_tau1_sub_const_ndpsi(mnl->w_fields[0], mnl->w_fields[1],
			     g_chi_up_spinor_field[j], g_chi_dn_spinor_field[j], 
			     -I*mnl->rat.mu[j], 1., mnl->EVMaxInv);
      
      /* Get the even parts X_j,e */
      /* H_eo_... includes tau_1 */
      H_eo_tm_ndpsi(mnl->w_fields[2], mnl->w_fields[3], 
		    g_chi_up_spinor_field[j], g_chi_dn_spinor_field[j], EO);
    }
    /* X_j,e^dagger \delta M_eo Y_j,o */
    deriv_Sb(EO, mnl->w_fields[2], mnl->w_fields[0], 
	     hf, mnl->rat.rmu[j]*mnl->forcefactor);
    deriv_Sb(EO, mnl->w_fields[3], mnl->w_fields[1],
	     hf, mnl->rat.rmu[j]*mnl->forcefactor);

    if(mnl->type == NDCLOVERRAT) {
      /* Get the even parts Y_j,e */
      H_eo_sw_ndpsi(mnl->w_fields[4], mnl->w_fields[5], 
		    mnl->w_fields[0], mnl->w_fields[1]);
    }
    else {
      /* Get the even parts Y_j,e */
      H_eo_tm_ndpsi(mnl->w_fields[4], mnl->w_fields[5], 
		    mnl->w_fields[0], mnl->w_fields[1], EO);

    }
    /* X_j,o \delta M_oe Y_j,e */
    deriv_Sb(OE, g_chi_up_spinor_field[j], mnl->w_fields[4], 
	     hf, mnl->rat.rmu[j]*mnl->forcefactor);
    deriv_Sb(OE, g_chi_dn_spinor_field[j], mnl->w_fields[5], 
	     hf, mnl->rat.rmu[j]*mnl->forcefactor);

    if(mnl->type == NDCLOVERRAT) {
      // even/even sites sandwiched by tau_1 gamma_5 Y_e and gamma_5 X_e
      sw_spinor(EE, mnl->w_fields[5], mnl->w_fields[2], 
		mnl->rat.rmu[j]*mnl->forcefactor);
      // odd/odd sites sandwiched by tau_1 gamma_5 Y_o and gamma_5 X_o
      sw_spinor(OO, g_chi_up_spinor_field[j], mnl->w_fields[1],
		mnl->rat.rmu[j]*mnl->forcefactor);
      
      // even/even sites sandwiched by tau_1 gamma_5 Y_e and gamma_5 X_e
      sw_spinor(EE, mnl->w_fields[4], mnl->w_fields[3], 
		mnl->rat.rmu[j]*mnl->forcefactor);
      // odd/odd sites sandwiched by tau_1 gamma_5 Y_o and gamma_5 X_o
      sw_spinor(OO, g_chi_dn_spinor_field[j], mnl->w_fields[0],
		mnl->rat.rmu[j]*mnl->forcefactor);
    }
  }
  // trlog part does not depend on the normalisation
  if(mnl->type == NDCLOVERRAT && mnl->trlog) {
    sw_deriv_nd(EE);
  }
  if(mnl->type == NDCLOVERRAT) {
    sw_all(hf, mnl->kappa, mnl->c_sw);
  }
  etime = gettime();
  if(g_debug_level > 1 && g_proc_id == 0) {
    printf("# Time for %s monomial derivative: %e s\n", mnl->name, etime-atime);
  }
  return;
}


void ndrat_heatbath(const int id, hamiltonian_field_t * const hf) {
  monomial * mnl = &monomial_list[id];
  solver_pm_t solver_pm;
  double atime, etime;
  atime = gettime();
  nd_set_global_parameter(mnl);
  mnl->iter1 = 0;
  if(mnl->type == NDCLOVERRAT) {
    init_sw_fields();
    sw_term((const su3**)hf->gaugefield, mnl->kappa, mnl->c_sw); 
    sw_invert_nd(mnl->mubar*mnl->mubar - mnl->epsbar*mnl->epsbar);
  }
  // we measure before the trajectory!
  if((mnl->rec_ev != 0) && (hf->traj_counter%mnl->rec_ev == 0)) {
    if(mnl->type != NDCLOVERRAT) phmc_compute_ev(hf->traj_counter-1, id, &Qtm_pm_ndbipsi);
    else phmc_compute_ev(hf->traj_counter-1, id, &Qsw_pm_ndbipsi);
  }

  // the Gaussian distributed random fields
  mnl->energy0 = 0.;
  random_spinor_field_eo(mnl->pf, mnl->rngrepro, RN_GAUSS);
  mnl->energy0 = square_norm(mnl->pf, VOLUME/2, 1);

  random_spinor_field_eo(mnl->pf2, mnl->rngrepro, RN_GAUSS);
  mnl->energy0 += square_norm(mnl->pf2, VOLUME/2, 1);
  if(mnl->decouple) {
    mnl->energy0 = 0;
  }
  // set solver parameters
  solver_pm.max_iter = mnl->maxiter;
  solver_pm.squared_solver_prec = mnl->accprec;
  solver_pm.no_shifts = mnl->rat.np;
  solver_pm.shifts = mnl->rat.nu;
  solver_pm.type = CGMMSND;
  solver_pm.M_ndpsi = &Qtm_pm_ndpsi;
  if(mnl->type == NDCLOVERRAT) solver_pm.M_ndpsi = &Qsw_pm_ndpsi;
  solver_pm.sdim = VOLUME/2;
  solver_pm.rel_prec = g_relative_precision_flag;
  mnl->iter0 = cg_mms_tm_nd(g_chi_up_spinor_field, g_chi_dn_spinor_field,
			     mnl->pf, mnl->pf2, &solver_pm);

  assign(mnl->w_fields[2], mnl->pf, VOLUME/2);
  assign(mnl->w_fields[3], mnl->pf2, VOLUME/2);

  // apply C to the random field to generate pseudo-fermion fields
  for(int j = (mnl->rat.np-1); j > -1; j--) {
    // Q_h * tau^1 - i nu_j
    // this needs phmc_Cpol = 1 to work!
    if(mnl->type == NDCLOVERRAT) {
      Qsw_tau1_sub_const_ndpsi(g_chi_up_spinor_field[mnl->rat.np], g_chi_dn_spinor_field[mnl->rat.np],
			       g_chi_up_spinor_field[j], g_chi_dn_spinor_field[j], 
			       I*mnl->rat.nu[j], 1., mnl->EVMaxInv);
    }
    else {
      Q_tau1_sub_const_ndpsi(g_chi_up_spinor_field[mnl->rat.np], g_chi_dn_spinor_field[mnl->rat.np],
			     g_chi_up_spinor_field[j], g_chi_dn_spinor_field[j], 
			     I*mnl->rat.nu[j], 1., mnl->EVMaxInv);
    }
    assign_add_mul(mnl->pf, g_chi_up_spinor_field[mnl->rat.np], I*mnl->rat.rnu[j], VOLUME/2);
    assign_add_mul(mnl->pf2, g_chi_dn_spinor_field[mnl->rat.np], I*mnl->rat.rnu[j], VOLUME/2);
  }

  etime = gettime();
  if(g_proc_id == 0) {
    if(g_debug_level > 1) {
      printf("# Time for %s monomial heatbath: %e s\n", mnl->name, etime-atime);
    }
    if(g_debug_level > 3) { 
      printf("called ndrat_heatbath for id %d energy %f\n", id, mnl->energy0);
    }
  }
  return;
}


double ndrat_acc(const int id, hamiltonian_field_t * const hf) {
  solver_pm_t solver_pm;
  monomial * mnl = &monomial_list[id];
  double atime, etime;
  atime = gettime();
  nd_set_global_parameter(mnl);
  if(mnl->type == NDCLOVERRAT) {
    sw_term((const su3**) hf->gaugefield, mnl->kappa, mnl->c_sw); 
    sw_invert_nd(mnl->mubar*mnl->mubar - mnl->epsbar*mnl->epsbar);
  }
  mnl->energy1 = 0.;

  solver_pm.max_iter = mnl->maxiter;
  solver_pm.squared_solver_prec = mnl->accprec;
  solver_pm.no_shifts = mnl->rat.np;
  solver_pm.shifts = mnl->rat.mu;
  solver_pm.type = CGMMSND;
  solver_pm.M_ndpsi = &Qtm_pm_ndpsi;
  if(mnl->type == NDCLOVERRAT) solver_pm.M_ndpsi = &Qsw_pm_ndpsi;
  solver_pm.sdim = VOLUME/2;
  solver_pm.rel_prec = g_relative_precision_flag;
  mnl->iter0 += cg_mms_tm_nd(g_chi_up_spinor_field, g_chi_dn_spinor_field,
			     mnl->pf, mnl->pf2,
			     &solver_pm);

  // apply R to the pseudo-fermion fields
  assign(mnl->w_fields[0], mnl->pf, VOLUME/2);
  assign(mnl->w_fields[1], mnl->pf2, VOLUME/2);
  for(int j = (mnl->rat.np-1); j > -1; j--) {
    assign_add_mul_r(mnl->w_fields[0], g_chi_up_spinor_field[j], 
		     mnl->rat.rmu[j], VOLUME/2);
    assign_add_mul_r(mnl->w_fields[1], g_chi_dn_spinor_field[j], 
		     mnl->rat.rmu[j], VOLUME/2);
  }

  mnl->energy1 = scalar_prod_r(mnl->pf, mnl->w_fields[0], VOLUME/2, 1);
  mnl->energy1 += scalar_prod_r(mnl->pf2, mnl->w_fields[1], VOLUME/2, 1);
  if(mnl->decouple) {
    mnl->energy1 = 0;
  }
  etime = gettime();
  if(g_proc_id == 0) {
    if(g_debug_level > 1) {
      printf("# Time for %s monomial acc step: %e s\n", mnl->name, etime-atime);
    }
    if(g_debug_level > 3) { // shoud be 3
      printf("called ndrat_acc for id %d dH = %1.10e\n", id, mnl->energy1 - mnl->energy0);
    }
  }
  return(mnl->energy1 - mnl->energy0);
}

double ndrat_energy(const int id, const su3** gaugefield) {
  solver_pm_t solver_pm;
  monomial * mnl = &monomial_list[id];
  double atime, etime;
  atime = gettime();

  double energy = 0;
  
  nd_set_global_parameter(mnl);
  if(mnl->type == NDCLOVERRAT) {
    sw_term((const su3**) gaugefield, mnl->kappa, mnl->c_sw);
    sw_invert_nd(mnl->mubar*mnl->mubar - mnl->epsbar*mnl->epsbar);
  }

  solver_pm.max_iter = mnl->maxiter;
  solver_pm.squared_solver_prec = mnl->accprec;
  solver_pm.no_shifts = mnl->rat.np;
  solver_pm.shifts = mnl->rat.mu;
  solver_pm.type = CGMMSND;
  solver_pm.M_ndpsi = &Qtm_pm_ndpsi;
  if(mnl->type == NDCLOVERRAT) solver_pm.M_ndpsi = &Qsw_pm_ndpsi;
  solver_pm.sdim = VOLUME/2;
  solver_pm.rel_prec = g_relative_precision_flag;
  int iter = cg_mms_tm_nd(g_chi_up_spinor_field, g_chi_dn_spinor_field,
               mnl->pf, mnl->pf2,
               &solver_pm);

  // apply R to the pseudo-fermion fields
  assign(mnl->w_fields[0], mnl->pf, VOLUME/2);
  assign(mnl->w_fields[1], mnl->pf2, VOLUME/2);
  for(int j = (mnl->rat.np-1); j > -1; j--) {
    assign_add_mul_r(mnl->w_fields[0], g_chi_up_spinor_field[j],
                     mnl->rat.rmu[j], VOLUME/2);
    assign_add_mul_r(mnl->w_fields[1], g_chi_dn_spinor_field[j],
                     mnl->rat.rmu[j], VOLUME/2);
  }

  energy = scalar_prod_r(mnl->pf, mnl->w_fields[0], VOLUME/2, 1);
  energy += scalar_prod_r(mnl->pf2, mnl->w_fields[1], VOLUME/2, 1);
  etime = gettime();
  if(g_proc_id == 0) {
    if(iter == -1) {
      printf("WARNING: solver for monomial %s in energy computation did not converge!\n",mnl->name);
    }
    if(g_debug_level > 1) {
      printf("# Time for %s monomial energy computation: %e s\n", mnl->name, etime-atime);
    }
    if(g_debug_level > 3) {
      printf("called ndrat_energy for id %d H_ndrat = %1.10e\n", id, energy);
    }
  }
  return(energy);
}

int init_ndrat_monomial(const int id) {
  monomial * mnl = &monomial_list[id];  

  mnl->EVMin = mnl->StildeMin / mnl->StildeMax;
  mnl->EVMax = 1.;
  mnl->EVMaxInv = 1./(sqrt(mnl->StildeMax));

  if(mnl->type == RAT || mnl->type == CLOVERRAT ||
     mnl->type == RATCOR || mnl->type == CLOVERRATCOR) {
    init_rational(&mnl->rat, 1);

    if(init_chi_spinor_field(VOLUMEPLUSRAND/2, (mnl->rat.np+2)/2) != 0) {
      fprintf(stderr, "Not enough memory for Chi fields! Aborting...\n");
      exit(0);
    }
  }
  else {
    init_rational(&mnl->rat, 0);
    mnl->EVMin = mnl->StildeMin / mnl->StildeMax;
    mnl->EVMax = 1.;
    mnl->EVMaxInv = 1./(sqrt(mnl->StildeMax));
    
    if(init_chi_spinor_field(VOLUMEPLUSRAND/2, (mnl->rat.np+1)) != 0) {
      fprintf(stderr, "Not enough memory for Chi fields! Aborting...\n");
      exit(0);
    }
  }

  return(0);
}

