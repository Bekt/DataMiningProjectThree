package ml.projectthree;

import ml.ColumnAttributes;
import ml.MLException;
import ml.Matrix;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import static java.lang.Math.log;

public class DecisionTree {

    DecisionTree left, right;
    String column, value, label;
    int k;

    public DecisionTree(Matrix features, Matrix labels, int k) {
        this.k = k;
        buildTree(features, labels);
    }

    public boolean isLeaf() {
        return left == null && right == null;
    }

    private void buildTree(Matrix features, Matrix labels) {

        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features and labels must have the same number of rows.");
        }

        int n = features.getNumRows();

        if (n <= k) {
            double index = labels.mostCommonValue(0);
            label = labels.getColumnAttributes(0).getValue((int) index);
        } else if (labels.isHomogeneous(0)) {
            double index = labels.getRow(0).get(0);
            label = labels.getColumnAttributes(0).getValue((int) index);
        } else {

            Matrix bestTrueFeatures = null, bestTrueLabels = null;
            Matrix bestFalseFeatures = null, bestFalseLabels = null;
            String columnName = "", valueName = "";
            double minInfo = Double.MAX_VALUE;

            for (int col = 0; col < features.getNumCols(); col++) {
                ColumnAttributes column = features.getColumnAttributes(col);

                for (int val = 0; val < column.size(); val++) {
                    Matrix trueFeatures = new Matrix(features, true),
                            trueLabels = new Matrix(labels, true);
                    Matrix falseFeatures = new Matrix(features, true),
                            falseLabels = new Matrix(labels, true);

                    Map<Double, Integer> trueLabelCount = new HashMap<Double, Integer>();
                    Map<Double, Integer> falseLabelCount = new HashMap<Double, Integer>();

                    for (int row = 0; row < n; row++) {
                        double featureVal = features.getRow(row).get(col);
                        double label = labels.getRow(row).get(0);
                        if ((int) featureVal == val) {
                            trueFeatures.addRow(features.getRow(row));
                            trueLabels.addRow(labels.getRow(row));
                            Integer count = trueLabelCount.get(label);
                            trueLabelCount.put(label, count == null ? 1 : count + 1);
                        } else {
                            falseFeatures.addRow(features.getRow(row));
                            falseLabels.addRow(labels.getRow(row));
                            Integer count = falseLabelCount.get(label);
                            falseLabelCount.put(label, count == null ? 1 : count + 1);
                        }
                    }

                    if (trueFeatures.getNumRows() == 0 || falseFeatures.getNumRows() == 0) {
                        continue;
                    }

                    double trueEntropy = getEntropy(trueLabelCount, trueFeatures.getNumRows());
                    double falseEntropy = getEntropy(falseLabelCount, falseFeatures.getNumRows());
                    double info = (trueEntropy * (trueFeatures.getNumRows() / (double) n))
                            + (falseEntropy * (falseFeatures.getNumRows() / (double) n));

                    if (info < minInfo) {
                        bestTrueFeatures = trueFeatures;
                        bestTrueLabels = trueLabels;
                        bestFalseFeatures = falseFeatures;
                        bestFalseLabels = falseLabels;
                        columnName = column.getName();
                        valueName = column.getValue(val);
                        minInfo = info;
                    }
                }
            }
            column = columnName;
            value = valueName;
            left = new DecisionTree(bestTrueFeatures, bestTrueLabels, k);
            right = new DecisionTree(bestFalseFeatures, bestFalseLabels, k);
        }
    }

    private static double getEntropy(Map<Double, Integer> labelsCount, int n) {
        double LOG_TWO = log(2);
        double entropy = 0.0;
        Set<Double> keys = labelsCount.keySet();
        for (double key : keys) {
            int count = labelsCount.get(key);
            double numer = count / (double) n;
            entropy += (-numer * (log(numer) / LOG_TWO));
        }
        return entropy;
    }

    public void prettyPrint(StringBuilder buffer, String parent) {
        for (int i = 0; i + 1 < buffer.length(); i++) {
            System.out.print(buffer.charAt(i));
        }
        System.out.println("|");
        for (int i = 0; i + 1 < buffer.length(); i++) {
            System.out.print(buffer.charAt(i));
        }

        if (isLeaf()) {
            System.out.println("+" + parent + "->'class'=" + label);
        } else {
            System.out.println("+" + parent + "->Is " + column + " == " + value + "?");
            buffer.append("   |");
            left.prettyPrint(buffer, "Yes");
            buffer.setCharAt(buffer.length() - 1, ' ');
            right.prettyPrint(buffer, "No");
            buffer.delete(buffer.length() - 4, buffer.length());
        }
    }

}
