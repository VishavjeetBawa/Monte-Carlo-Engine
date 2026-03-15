#include <math.h>

__device__
void brownian_bridge(
    double* z,
    double* w,
    int N,
    double dt)
{
    w[N-1] = sqrt(N*dt) * z[0];

    int left = 0;
    int right = N-1;
    int dim = 1;

    while(dim < N)
    {
        int mid = (left + right)/2;

        double t_left = left*dt;
        double t_mid = mid*dt;
        double t_right = right*dt;

        double mean =
            ((t_right - t_mid)*w[left] +
             (t_mid - t_left)*w[right]) /
            (t_right - t_left);

        double var =
            (t_mid - t_left) *
            (t_right - t_mid) /
            (t_right - t_left);

        w[mid] = mean + sqrt(var)*z[dim];

        dim++;
        right = mid;
    }

    for(int i=N-1;i>0;i--)
        w[i] = w[i]-w[i-1];
}
