package com.jujutsu.tsne;

import org.ejml.data.DenseMatrix64F;

import java.io.*;
import java.util.Arrays;

import static com.jujutsu.utils.EjmlOps.repmat;
import static org.ejml.ops.CommonOps.*;
import static org.ejml.ops.MatrixFeatures.hasNaN;

/**
 * Created by jjensen on 08.07.2016.
 */
public class KernelMapping {
    private DenseMatrix64F A, sig_nb;

    public KernelMapping(DenseMatrix64F a, DenseMatrix64F sig_nb) {
        A = a;
        this.sig_nb = sig_nb;
    }

    public KernelMapping() {
    }

    public void loadKernel(String filepath) {
        try {
            FileInputStream in = new FileInputStream(filepath);
            ObjectInputStream s = new ObjectInputStream(in);
            this.A = (DenseMatrix64F) s.readObject();
            this.sig_nb = (DenseMatrix64F) s.readObject();
        } catch (ClassNotFoundException | IOException e) {
            e.printStackTrace();
        }
    }

    public void kmapTrain(double[][] x_dist, double[][] y, double sign, String filepath) {
        kmapTrain(new DenseMatrix64F(x_dist), new DenseMatrix64F(y), sign, filepath);
    }


    public void kmapTrain(DenseMatrix64F x_dist, DenseMatrix64F y, double sign, String filepath) {
        int k_nb = 10;      //number of neighbours
        int n = x_dist.getNumCols();
        boolean f_local = true;    //false for global and true for local widths

        // compute widths
        DenseMatrix64F temp = new DenseMatrix64F(n, n);
        DenseMatrix64F kernel = new DenseMatrix64F(n, n);
        DenseMatrix64F inv_kernel = new DenseMatrix64F(n, n);
        DenseMatrix64F sum = new DenseMatrix64F(n, 1);
        DenseMatrix64F a = new DenseMatrix64F(n, y.numCols);

        DenseMatrix64F sig_nb = determine_sigma_local(x_dist, k_nb);
        scale(sign, sig_nb);
        DenseMatrix64F sig_nb_rep = repmat(sig_nb, n, 1);
        elementDiv(x_dist, sig_nb_rep, temp);
        scale(-0.5, temp);
        elementExp(temp, kernel);

        sumRows(kernel, sum);
        sum = repmat(sum, 1, n);
        elementDiv(kernel, sum);
        pinv(kernel, inv_kernel);
        mult(inv_kernel, y, a);
        try {
            this.A = a;
            this.sig_nb = sig_nb;
            FileOutputStream f = new FileOutputStream(filepath);
            ObjectOutput s = new ObjectOutputStream(f);
            s.writeObject(a);
            s.writeObject(sig_nb);
            s.flush();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public DenseMatrix64F kmap(double[][] x_dist_ose) {
        return kmap(new DenseMatrix64F(x_dist_ose));
    }

    public DenseMatrix64F kmap(DenseMatrix64F x_dist_ose) {
        int rows = x_dist_ose.getNumRows();
        int cols = x_dist_ose.getNumCols();
        DenseMatrix64F temp = new DenseMatrix64F(rows, cols);
        DenseMatrix64F kernel_ose = new DenseMatrix64F(rows, cols);
        DenseMatrix64F sum = new DenseMatrix64F(rows, 1);
        DenseMatrix64F y_ose = new DenseMatrix64F(rows, this.A.getNumCols());

        DenseMatrix64F sig_nb_rep = repmat(this.sig_nb, rows, 1);
        elementDiv(x_dist_ose, sig_nb_rep, temp);
        scale(-0.5, temp);
        elementExp(temp, kernel_ose);

        sumRows(kernel_ose, sum);
        sum = repmat(sum, 1, cols);
        elementDiv(kernel_ose, sum);
        mult(kernel_ose, this.A, y_ose);
        if (hasNaN(y_ose)) {
            System.err.println("NaN elements in y_ose. Consider using larger sigma for training.");
        }
        return y_ose;
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
            sig_nb.set(0, i, sorted[j - 1] * sorted[j - 1]);
        }

        return sig_nb;
    }

    public int getDimension(){
        return this.A.getNumCols();
    }

}
