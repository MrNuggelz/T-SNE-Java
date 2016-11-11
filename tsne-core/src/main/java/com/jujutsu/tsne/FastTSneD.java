package com.jujutsu.tsne;

import static com.jujutsu.utils.MatrixOps.*;
import static com.jujutsu.utils.MatrixOps.scalarInverse;
import static com.jujutsu.utils.MatrixOps.sqrt;

/**
 * Created by jjensen on 11.11.2016.
 */
public class FastTSneD extends FastTSne {

    @Override
    public double[][] tsne(double[][] X, int k, int initial_dims, double perplexity) {
        return super.tsne(X, k, initial_dims, perplexity, 2000, false);
    }

    @Override
    public R x2p(double[][] D, double tol, double perplexity) {
        int n               = D.length;
        // D seems correct at this point compared to Python version
        double [][] P       = fillMatrix(n,n,0.0);
        double [] beta      = fillMatrix(n,n,1.0)[0];
        double logU         = Math.log(perplexity);
        System.out.println("Starting x2p...");
        for (int i = 0; i < n; i++) {
            if (i % 500 == 0)
                System.out.println("Computing P-values for point " + i + " of " + n + "...");
            double betamin = Double.NEGATIVE_INFINITY;
            double betamax = Double.POSITIVE_INFINITY;
            double [][] Di = getValuesFromRow(D, i,concatenate(range(0,i),range(i+1,n)));

            R hbeta = Hbeta(Di, beta[i]);
            double H = hbeta.H;
            double [][] thisP = hbeta.P;

            // Evaluate whether the perplexity is within tolerance
            double Hdiff = H - logU;
            int tries = 0;
            while(Math.abs(Hdiff) > tol && tries < 50){
                if (Hdiff > 0){
                    betamin = beta[i];
                    if (Double.isInfinite(betamax))
                        beta[i] = beta[i] * 2;
                    else
                        beta[i] = (beta[i] + betamax) / 2;
                } else{
                    betamax = beta[i];
                    if (Double.isInfinite(betamin))
                        beta[i] = beta[i] / 2;
                    else
                        beta[i] = ( beta[i] + betamin) / 2;
                }

                hbeta = Hbeta(Di, beta[i]);
                H = hbeta.H;
                thisP = hbeta.P;
                Hdiff = H - logU;
                tries = tries + 1;
            }
            assignValuesToRow(P, i,concatenate(range(0,i),range(i+1,n)),thisP[0]);
        }

        R r = new R();
        r.P = P;
        r.beta = beta;
        double sigma = mean(sqrt(scalarInverse(beta)));

        System.out.println("Mean value of sigma: " + sigma);

        return r;
    }
}
