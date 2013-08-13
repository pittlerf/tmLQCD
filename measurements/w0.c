/***********************************************************************
 * Copyright (C) 2013 Albert Deuzeman
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

#include <measurements/w0.h>

typedef su3

void w0_measurement(const int traj, const int id, const int ieo)
{
  double const ref_value = 0.30;
  double t0;
  double w0;
  double ed;
  
  double epsilon = measurement_list[max_no_measurements].epsilon;
  gradient_control *control = construct_gradient_control(epsilon, 1.0 /* starting guess for distance */);
  gradient_smear(control, g_gf);
  
  double hist_t[3] = {0.0, 0.0, 0.0};
  double hist_t2_ed[3] = {0.0, 0.0, 0.0};
  gauge_field_t hist_field[3];
  
  for (int ctr = 0; ctr < 3; ++ctr)
  {
    gradient_smear(control, g_gf);
    hist_field[ctr] = get_gauge_field();
    hist_t[ctr] = control->distance;
    hist_t2_ed[ctr] = control->distance * control->distance * energy_density(control->result);
    swap_gauge_field(&hist_field[ctr], &control->U[1]);
  }345

  
  
  hist_t[1] = control->distance;
  hist_t2_ed[1] = control->distance * control->distance * energy_density(control->result);
  swahist_field[1] = control->result;
  
  /* Run Newton's algorithm to determine t0 first. W0 will need some derivatives, so more measurements.
     Those measurements will be available after t0 has been calculated. */
  double deriv_t2_ed  = (hist_t2_ed[1] - hist_t2_ed[0]) / (hist_t[1] - hist_t[0]);
  double new_t = hist_t[1] + ((ref_value - hist_t2_ed[1]) / deriv_t2_ed);
  

  
  if( g_proc_id == 0 )
  {
    FILE *outfile;
    char filename[] = "w0.data";
    outfile = fopen(filename, "a");

    if( outfile == NULL )
    {
      char error_message[200];
      snprintf(error_message,200,"Couldn't open %s for appending during measurement %d!", filename, id);
      fatal_error(error_message, "w0_measurement");
    }
    fprintf(outfile, "traj %.8d: w0 = %14.12lf, t0 = %14.12lf\n", traj, w0, t0);
    fclose(outfile);
  }

  return;
}

#endif
