#include "gradient.ih"

void free_gradient_control(gradient_control *control)
{
  if (!control)
    return;

  return_gauge_field(&control->U[1]); /* All other fields are shallow copies. */
  free(control);
  
  return;

}
