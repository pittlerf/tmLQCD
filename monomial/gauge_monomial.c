/***********************************************************************
 *
 * Copyright (C) 2008 Carsten Urbach
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
#ifdef OMP 
# include <omp.h>
#endif
#include "global.h"
#include "su3.h"
#include "su3adj.h"
#include "ranlxd.h"
#include "sse.h"
#include "start.h"
#include "gettime.h"
#include "get_rectangle_staples.h"
#include "gamma.h"
#include "get_staples.h"
#include "read_input.h"
#include "measure_gauge_action.h"
#include "measure_rectangles.h"
#include "monomial/monomial.h"
#include "hamiltonian_field.h"
#include "gauge_monomial.h"
#include "dirty_shameful_business.h"
#include "expo.h"

void gauge_derivative_analytical(const int id, hamiltonian_field_t * const hf) {
  monomial * mnl = &monomial_list[id];
    double atime, etime;
    atime = gettime();
  #ifdef OMP
  #pragma omp parallel
    {
  #endif

    su3 ALIGN v, w;
    int i, mu;
    su3 *z;
    su3adj *xm;
    double factor = -1. * g_beta/3.0;

    if(mnl->use_rectangles) {
      mnl->forcefactor = 1.;
      factor = -mnl->c0 * g_beta/3.0;
    }
    
  #ifdef OMP
  #pragma omp for
  #endif
    for(i = 0; i < VOLUME; i++) { 
      for(mu=0;mu<4;mu++) {
        z=&hf->gaugefield[i][mu];
        xm=&hf->derivative[i][mu];
        get_staples(&v,i,mu, (const su3**) hf->gaugefield); 
        _su3_times_su3d(w,*z,v);
        _trace_lambda_mul_add_assign((*xm), factor, w);
        
        if(mnl->use_rectangles) {
	  get_rectangle_staples(&v, i, mu);
	  _su3_times_su3d(w, *z, v);
	  _trace_lambda_mul_add_assign((*xm), factor*mnl->c1/mnl->c0, w);
        }
      }
    }

  #ifdef OMP
    } /* OpenMP closing brace */
  #endif
    etime = gettime();
    if(g_debug_level > 1 && g_proc_id == 0) {
      printf("# Time for analytical %s monomial derivative: %e s\n", mnl->name, etime-atime);
    }
    return;
}

/* this function calculates the derivative of the momenta: equation 13 of Gottlieb */
void gauge_derivative(const int id, hamiltonian_field_t * const hf) {
    monomial * mnl = &monomial_list[id];
  if(0) {
    double atime, etime;
    atime = gettime();
  #ifdef OMP
  #pragma omp parallel
    {
  #endif

    su3 ALIGN v, w;
    int i, mu;
    su3 *z;
    su3adj *xm;
    double factor = -1. * g_beta/3.0;

    if(mnl->use_rectangles) {
      mnl->forcefactor = 1.;
      factor = -mnl->c0 * g_beta/3.0;
    }
    
  #ifdef OMP
  #pragma omp for
  #endif
    for(i = 0; i < VOLUME; i++) { 
      for(mu=0;mu<4;mu++) {
        z=&hf->gaugefield[i][mu];
        xm=&hf->derivative[i][mu];
        get_staples(&v,i,mu, (const su3**) hf->gaugefield); 
        _su3_times_su3d(w,*z,v);
        _trace_lambda_mul_add_assign((*xm), factor, w);
        
        if(mnl->use_rectangles) {
	  get_rectangle_staples(&v, i, mu);
	  _su3_times_su3d(w, *z, v);
	  _trace_lambda_mul_add_assign((*xm), factor*mnl->c1/mnl->c0, w);
        }
      }
    }

  #ifdef OMP
    } /* OpenMP closing brace */
  #endif
    etime = gettime();
    if(g_debug_level > 1 && g_proc_id == 0) {
      printf("# Time for %s monomial derivative: %e s\n", mnl->name, etime-atime);
    }
    return;
  } else {
    double atime = gettime();
    /* Get some memory set aside for gauge fields and copy our current field */
    gauge_field_t rotated[2];
    rotated[0] = get_gauge_field();
    rotated[1] = get_gauge_field();

    memmove(rotated[0], g_gf, sizeof(su3_tuple) * (VOLUMEPLUSRAND + g_dbw2rand) + 1);
    memmove(rotated[1], g_gf, sizeof(su3_tuple) * (VOLUMEPLUSRAND + g_dbw2rand) + 1);

    su3adj rotation;
    double *ar_rotation = (double*)&rotation;
    double const eps = 5e-6;
    double const epsilon[2] = {-eps,eps};
    su3 old_value;
    su3 mat_rotation;
    double* xm;
    su3* link;

    stout_control* control = construct_stout_control(1,1,0.18);

    for(int x = 0; x < VOLUME; ++x)
    {
      for(int mu = 0; mu < 4; ++mu)
      {
        xm=(double*)&hf->derivative[x][mu];
        control->smearing_performed = 0;
        for (int component = 0; component < 8; ++component)
        {
          double h_rotated[2] = {0.0,0.0};
          //printf("Rotating at %d %d in component %d\n",x,mu,component);
          for(int direction = 0; direction < 2; ++direction) 
          {
            link=&rotated[direction][x][mu];
            // save current value of gauge field
            memmove(&old_value, link, sizeof(su3));
            /* Introduce a rotation along one of the components */
            memset(ar_rotation, 0, sizeof(su3adj));
            ar_rotation[component] = epsilon[direction];
            exposu3(&mat_rotation, &rotation);
            _su3_times_su3(rotated[direction][x][mu], mat_rotation, old_value);

            //if(x == 0 && mu == 0 && direction == 0 && component == 0)
            //  print_su3(&rotated[0][0][0]);
              
            stout_smear(control, rotated[direction]);
 
            //if(x == 1 && mu == 1 && direction == 1 && component == 1 ) {
            //  stout_smear_forces(control,df);
            //  fprintf(stderr, "[DEBUG] Comparison of force calculation at [1][1]!\n");
            //  fprintf(stderr, "   smear forces <-> numerical total force\n");
            //  fprintf(stderr, "    [%d]  %+14.12f <-> ", component, control->force_result[1][1].d1); //*/
            //}

            //if(x == 0 && mu == 0 && direction == 0 && component == 0)
            //  print_su3(&control->result[0][0]);

            //su3 test;
            //if(x == 0 && mu == 0 && direction == 0 && component == 0) {
            //  restoresu3(&test,&control->result[0][0]);
            //  print_su3(&test);
            //}
 
            // compute gauge action
            g_update_gauge_energy = 1;
            g_update_rectangle_energy = 1;
            h_rotated[direction] = -1.0*g_beta*measure_gauge_action(&control->result[0][0]);
            // reset modified part of gauge field
            memmove(link,&old_value, sizeof(su3));
          } // direction
          // calculate force contribution from gauge field due to rotation
          xm[component] += (h_rotated[1]-h_rotated[0])/(2*eps);
          //if( x == 1 && mu == 1 && component == 1 )
          //  fprintf(stderr, "%+14.12f\n", df[1][1].d1); //*/
        } // component
      } // mu
    } // x
    free_stout_control(control);
    return_gauge_field(&rotated[0]);
    return_gauge_field(&rotated[1]);
    double etime = gettime();
    if(g_debug_level > 1 && g_proc_id == 0) {
      printf("# Time for numerical %s monomial derivative: %e s\n", mnl->name, etime-atime);
    }
    g_update_gauge_energy = 1;
    g_update_rectangle_energy = 1;
    return;
  }
}

void gauge_heatbath(const int id, hamiltonian_field_t * const hf) {
  monomial * mnl = &monomial_list[id];
  
  if(mnl->use_rectangles) mnl->c0 = 1. - 8.*mnl->c1;
  
  mnl->energy0 = g_beta*(mnl->c0 * measure_gauge_action(_AS_GAUGE_FIELD_T(hf->gaugefield)));

  if(mnl->use_rectangles) {
    mnl->energy0 += g_beta*(mnl->c1 * measure_rectangles( (const su3**) hf->gaugefield));
  }
  if(g_proc_id == 0 && g_debug_level > 3) {
    printf("called gauge_heatbath for id %d %d energy0 = %lf\n", id, mnl->even_odd_flag, mnl->energy0);
  }
}

double gauge_acc(const int id, hamiltonian_field_t * const hf) {
  monomial * mnl = &monomial_list[id];
  
  mnl->energy1 = g_beta*(mnl->c0 * measure_gauge_action( _AS_GAUGE_FIELD_T(hf->gaugefield)));
  if(mnl->use_rectangles) {
    mnl->energy1 += g_beta*(mnl->c1 * measure_rectangles( (const su3**) hf->gaugefield));
    }
  if(g_proc_id == 0 && g_debug_level > 3) {
    printf("called gauge_acc for id %d %d dH = %1.10e\n", 
	   id, mnl->even_odd_flag, mnl->energy0 - mnl->energy1);
  }
  return(mnl->energy0 - mnl->energy1);
}
