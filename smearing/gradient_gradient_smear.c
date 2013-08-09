#include "gradient.ih"

void gradient_smear(gradient_control *control, gauge_field_t in)
{
  /* Allocate the required memory */
  control->U[0] = in;
  gauge_field_t Z01 = get_gauge_field();
  gauge_field_t buffer = get_gauge_field();
  
  su3 ALIGN tmp1;
  su3 ALIGN tmp2;
  
  /* Calculate the iteration number */
  unsigned int steps = (int)ceil(control->distance / control->epsilon);
  int remainder = ((steps - (control->distance / control->epsilon)) > 1e-6);

  // Calculate coefficients for Runge-Kutta
  double e14    =   1.0 * control->epsilon /  4.0;
  double e34    =   3.0 * control->epsilon /  4.0;
  double e1736  = -17.0 * control->epsilon / 36.0;
  double e89    =   8.0 * control->epsilon /  9.0;

  // Start of the gradient flow */
#pragma omp parallel private(staples, tmp)
  for(unsigned int iter = 0; iter < steps; ++iter)
  {    
    if (remainder && iter == (steps - 1))
    {
      double rescale = (control->distance / control->epsilon) - iter;
      e14   *= rescale;
      e34   *= rescale;
      e1736 *= rescale;
      e89   *= rescale;
    }
  
    // Runge-Kutta step 1.
#pragma omp for
    for (unsigned int x = 0; x < VOLUME; ++x)
      for (unsigned int mu = 0; mu < 4; ++mu)
      {
        // Note the difference with Luescher's definition (0907.5491v3 p14):
        // The orientation of the plaquette is inverted there and a minus sign is added.
        // Since we project to the traceless antihermitian vector space, 
        // we can leave out the minus sign.
        // This brings it in line with our implementation of stout smearing, coincidentally.
        
        // Note also that we're not updating links one by one.
        // This would be impossible to parallelize, but our implementation will differ
        // from Luescher's by discretization artifacts of hard to determine size.
        
        // The Runge-Kutta implementation follows (1006.4518v2, p19f).
        
        generic_staples(&tmp1, x, mu, in);
        _su3_times_su3d(tmp2, tmp1, in[x][mu]);
        project_traceless_antiherm(&tmp2);        
        _real_times_su3(tmp1, e14, tmp2);
        exposu3_in_place(&tmp1);
        _su3_times_su3(buffer[x][mu], tmp1, in[x][mu]);
        _real_times_su3(Z01[x][mu], e1736, tmp2); // Prepare for next step of Runge-Kutta.
      }

    #pragma omp single
    {
      swap_gauge_field(&control->U[1], &buffer);
      exchange_gauge_field(&control->U[1]); /* The edge terms will have changed by now */
      in = control->U[1]; /* Shallow copy intended */
    }
    
    // Runge-Kutta step 2.
#pragma omp for
    for (unsigned int x = 0; x < VOLUME; ++x)
      for (unsigned int mu = 0; mu < 4; ++mu)
      {
        generic_staples(&tmp1, x, mu, in);
        _su3_times_su3d(tmp2, tmp1, in[x][mu]);
        project_traceless_antiherm(&tmp2);        
        _real_times_su3(tmp2, e89, tmp2);
        _su3_plus_su3(Z01[x][mu], tmp2, Z01[x][mu]);
        exposu3_copy(&tmp1, &Z01[x][mu]);
        _su3_times_su3(buffer[x][mu], tmp1, in[x][mu]);
      }

    #pragma omp single
    {
      swap_gauge_field(&control->U[1], &buffer);
      exchange_gauge_field(&control->U[1]); /* The edge terms will have changed by now */
      in = control->U[1]; /* Shallow copy intended */
    }
    
    // Runge-Kutta step 3.
#pragma omp for
    for (unsigned int x = 0; x < VOLUME; ++x)
      for (unsigned int mu = 0; mu < 4; ++mu)
      {
        generic_staples(&tmp1, x, mu, in);
        _su3_times_su3d(tmp2, tmp1, in[x][mu]);
        project_traceless_antiherm(&tmp2);        
        _real_times_su3(tmp2, e34, tmp2);
        _su3_minus_su3(tmp1, tmp2, Z01[x][mu]);
        exposu3_in_place(&tmp1);
        _su3_times_su3(buffer[x][mu], tmp1, in[x][mu]);
      }

    #pragma omp single
    {
      swap_gauge_field(&control->U[1], &buffer);
      exchange_gauge_field(&control->U[1]); /* The edge terms will have changed by now */
      in = control->U[1]; /* Shallow copy intended */
    }
  }

  control->result = control->U[1];
  return_gauge_field(&buffer);
  return_gauge_field(&Z01);
}
