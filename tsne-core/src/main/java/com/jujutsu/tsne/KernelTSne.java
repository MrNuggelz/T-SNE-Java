package com.jujutsu.tsne;

import org.ejml.data.DenseMatrix64F;

import java.io.*;
import java.util.Arrays;
import java.util.Objects;

import static com.jujutsu.utils.EjmlOps.repmat;
import static org.ejml.ops.CommonOps.*;
import static org.ejml.ops.MatrixFeatures.hasNaN;

/**
 * Created by jjensen on 08.07.2016.
 */
public class KernelTSne {
    private final TSne tsne;
    private DenseMatrix64F A, sig_nb;

    public KernelTSne(TSne tsne){
        this.tsne = tsne;
    }

    public KernelTSne(TSne tsne, String filepath){
        this.tsne = tsne;
        try {
            FileInputStream in = new FileInputStream(filepath);
            ObjectInputStream s = new ObjectInputStream(in);
            this.A = (DenseMatrix64F)s.readObject();
            this.sig_nb = (DenseMatrix64F)s.readObject();
        } catch (ClassNotFoundException | IOException e) {
            e.printStackTrace();
        }
    }

    public void kmapTrain(double[][] x, String filepath) {
        DenseMatrix64F y = new DenseMatrix64F(this.tsne.tsne(x, 2, 5, 20.0));
        DenseMatrix64F x_dist = new DenseMatrix64F(x);

        int k_nb = 2;      //number of neighbours
        int n = x_dist.getNumCols();
        boolean f_local = true;    //false for global and true for local widths

        // compute widths
        DenseMatrix64F temp = new DenseMatrix64F(n, n);
        DenseMatrix64F kernel = new DenseMatrix64F(n, n);
        DenseMatrix64F inv_kernel = new DenseMatrix64F(n, n);
        DenseMatrix64F sum = new DenseMatrix64F(n, 1);
        DenseMatrix64F a = new DenseMatrix64F(n, y.numCols);

        DenseMatrix64F sig_nb = determine_sigma_local(x_dist, k_nb);
        DenseMatrix64F sig_nb_rep = repmat(sig_nb, n, 1);
        elementDiv(x_dist, sig_nb_rep, temp);
        scale(-0.5, temp);
        elementExp(temp, kernel);

        sumRows(kernel, sum);
        sum = repmat(sum, 1, n);
        elementDiv(kernel, sum);
        pinv(kernel, inv_kernel);
        mult(inv_kernel, y, a);
        if (!Objects.equals(filepath, "")){
            try {
                FileOutputStream f = new FileOutputStream(filepath);
                ObjectOutput s = new ObjectOutputStream(f);
                s.writeObject(a);
                s.writeObject(sig_nb);
                s.flush();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public void kmapTest(double[][] x){
        int n = this.sig_nb.getNumCols();
        DenseMatrix64F x_dist_ose = new DenseMatrix64F(x);
        DenseMatrix64F temp = new DenseMatrix64F(n, n);
        DenseMatrix64F kernel_ose = new DenseMatrix64F(n, n);
        DenseMatrix64F sum = new DenseMatrix64F(n, 1);
        DenseMatrix64F y_ose = new DenseMatrix64F(n, this.A.getNumCols());

        DenseMatrix64F sig_nb_rep = repmat(this.sig_nb, n, 1);
        x_dist_ose = repmat(x_dist_ose, n, 1);
        elementDiv(x_dist_ose, sig_nb_rep, temp);
        scale(-0.5, temp);
        elementExp(temp, kernel_ose);

        sumRows(kernel_ose, sum);
        sum = repmat(sum, 1, n);
        elementDiv(kernel_ose, sum);
        mult(kernel_ose,this.A,y_ose);
        if (hasNaN(y_ose)){
            System.err.println("NaN elements in y_ose. Consider using larger sigma for training.");
        }
    }

    private static DenseMatrix64F determine_sigma_local(DenseMatrix64F dist, int k_nb) {
        int n = dist.getNumCols();
        DenseMatrix64F sig_nb = new DenseMatrix64F(1, n);
        for (int i = 0; i < n; i++) {
            double[] sorted = new double[n];
            System.arraycopy(dist.getData(), n * i, sorted, 0, n);
            Arrays.sort(sorted);
            int j = (int) Arrays.stream(sorted).filter(d -> d == 0).count() + k_nb;
            if (j >= n) {
                System.err.println("k_nb to large");
                System.exit(-1);
            }
            sig_nb.set(0, i, sorted[j] * sorted[j] * 0.1);
        }

        return sig_nb;
    }

}
