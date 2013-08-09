#include "gradient.ih"

void gradient_smear(gradient_control *control, gauge_field_t in)
{
  /* Allocate the required memory */
  control->U[0] = in;
  gauge_field_t buffer = get_gauge_field();
  su3 ALIGN staples;
  su3 ALIGN tmp;
  
  /* Calculate the iteration number */
  unsigned int steps = (int)floor(control->distance / control->epsilon);
  double red_eps = control->distance - (steps * control->epsilon);
  int remainder = (red_eps / epsilon) > 1e-6;
  
  /* start of the the stout smearing **/
#pragma omp parallel private(staples, tmp)
  for(unsigned int iter = 0; iter < steps; ++iter)
  {
#pragma omp for
    for (unsigned int x = 0; x < VOLUME; ++x)
      for (unsigned int mu = 0; mu < 4; ++mu)
      {
        // Note the difference with Luescher's definition (0907.5491v3 p14):
        // The orientation of the plaquette is inverted there and a minus sign is added.
        // Since we project to the traceless antihermitian vector space, 
        // we can leave out the minus sign.
        // This brings it in line with our implementation of stout smearing, coincidentally.
        generic_staples(&staples, x, mu, in);
        _su3_times_su3d(tmp, staples, in[x][mu]);
        project_traceless_antiherm(&tmp);        
        _real_times_su3(tmp, control->epsilon, tmp);
        exposu3_in_place(&tmp);
        _su3_times_su3(buffer[x][mu], tmp, in[x][mu]);
      }

    /* Prepare for the next iteration, swap and exchange fields */
    /* There should be an implicit OMP barrier here */

    #pragma omp single
    {
      swap_gauge_field(&control->U[1], &buffer);
      exchange_gauge_field(&control->U[1]); /* The edge terms will have changed by now */
      in = control->U[1]; /* Shallow copy intended */
    }
  }

  control->result = control->U[1];
  return_gauge_field(&buffer);
}


  
