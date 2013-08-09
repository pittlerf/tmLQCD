#include "gradient.ih"

gradient_control *construct_gradient_control(double epsilon, double distance)
{
  gradient_control *control = (gradient_control*)malloc(sizeof(gradient_control));
  control->epsilon = epsilon;
  control->distance = distance;
  
  control->U[1] = get_gauge_field();
  
  control->current_distance = 0.0; /* Tracks the distance already integrated over */
  
  control->result = NULL; /* Set after calculation */
  
  return control;
}
