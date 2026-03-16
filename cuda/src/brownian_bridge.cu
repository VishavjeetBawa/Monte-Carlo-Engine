#include <math.h>

__device__
void brownian_bridge(
        double* z,
        double* w,
        int N,
        double dt)
{
    // terminal time
    w[N-1] = sqrt(N*dt) * z[0];

    int left[512];
    int right[512];

    int top = 0;

    left[top] = 0;
    right[top] = N-1;

    int dim = 1;

    while(top >= 0 && dim < N)
    {
        int l = left[top];
        int r = right[top];
        top--;

        if(r-l <= 1)
            continue;

        int m = (l+r)/2;

        double t_l = l*dt;
        double t_r = r*dt;
        double t_m = m*dt;

        double mean =
            ((t_r-t_m)*w[l] +
             (t_m-t_l)*w[r]) /
            (t_r-t_l);

        double var =
            (t_m-t_l)*(t_r-t_m)/(t_r-t_l);

        w[m] = mean + sqrt(var)*z[dim];

        dim++;

        top++;
        left[top]=l;
        right[top]=m;

        top++;
        left[top]=m;
        right[top]=r;
    }

    // convert bridge path → increments
    for(int i=N-1;i>0;i--)
        w[i] -= w[i-1];
}
