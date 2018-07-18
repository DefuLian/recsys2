#include <string.h>
#include <math.h>
#include <mex.h>
/* using cyclic coordinate descent for optimizing */
/* binary quadratic problem: min x' A x - 2 b' x, s.t. x in {+1,-1}^k */
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    double *x;/*variable of size r x 1*/
    double *A;/*coefficient matrix of size r x r */
    double *b;/*vector of size r x 1 */
    double maxItr;
    mwSize r;
   
    mwSize i,k,it,no_change_count;
    double ss, total_val, alpha = 0;
    bool converge = false;
    x = mxGetPr(prhs[0]);
    A = mxGetPr(prhs[1]);
    b = mxGetPr(prhs[2]);
    maxItr = *(mxGetPr(prhs[3]));
    if(nrhs == 5){
		alpha = *mxGetPr(prhs[4]);
	}
	
    r = mxGetN(prhs[1]);
    
    plhs[0] = mxCreateDoubleMatrix(r, 1, mxREAL);
	double *x_new = mxGetPr(plhs[0]);
    memcpy(x_new, x, r * sizeof(double));
    it = 0;
    
    while (!converge){
        total_val = 0 ;
        no_change_count = 0;
        for (k = 0; k < r; k ++){
            ss = 0;
            for (i = 0; i < r; i ++)
                if (i != k)
                    ss += A[k+i*r]*x_new[i];
            /*mexPrintf("%d,%f,",k,ss);*/
            ss -= b[k];
            /*mexPrintf("%f,", ss);*/
            if(nrhs < 5){
                if (ss > 1e-6){
                    if (x_new[k] == -1)
                        no_change_count ++;
                    else
                        x_new[k] = -1;
                }
                else if (ss < -1e-6){
                    if (x_new[k] == 1)
                        no_change_count ++;
                    else
                        x_new[k] = 1;
                }
                else{
                    no_change_count ++;
                }
            }
            else
            {
                if(A[k+k*r]>1e-16){
                    total_val += fabs(x_new[k] + ss / A[k+k*r]);
                    x_new[k] = - ss / (A[k+k*r]+alpha);
                }
            }
                
        }
        /*mexPrintf("\n");*/
        if ((it >= (int)maxItr-1) || (nrhs==5?(total_val<1e-4):(no_change_count == r)))
            converge = true;
        it ++;
    }
}
    
