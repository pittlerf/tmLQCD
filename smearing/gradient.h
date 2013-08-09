#ifndef GUARD_SMEARING_GRADIENT_H
#define GUARD_SMEARING_GRADIENT_H

#include <buffers/adjoint.h>
#include <buffers/gauge.h>
#include <buffers/utils.h>

#include <smearing/utils.h>

typedef struct
{ 
  /* Parameters */
  double epsilon; /* Step size in the Runge-Kutta integration */
  double distance;
 
  gauge_field_t U[2];
  
  double current_distance /* Tracks the state of the integrator */
  
  /* Final result */
  gauge_field_t result; /* Set upon calculation */
} gradient_control;


gradient_control *construct_gradient_control(int calculate_force_terms, double epsilon, double distance);
void free_gradient_control(gradient_control *control);

void gradient_smear(gradient_control *control, gauge_field_t in);

#endif