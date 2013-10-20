package ml.projectthree;

import helpers.Rand;
import ml.ColumnAttributes;
import ml.ColumnAttributes.ColumnType;
import ml.MLException;
import ml.Matrix;

import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

import static java.lang.Math.log;
import static helpers.Vector.sampleWithReplacement;


/**
 * Note to future self: I know this code is shit, but I was on a deadline, sorry!
 */
public class DecisionTree {

    DecisionTree left, right;
    Split splitInfo;
    String label;
    double labelValue;
    int k;

    class Split {
        ColumnType columnType;
        String columnName;
        String columnValue;
        int columnIndex;
        double columnMean;
    }

    public DecisionTree(int k) {
        this.k = k;
    }

    public boolean isLeaf() {
        return left == null && right == null;
    }

    public void buildEntropyTree(Matrix features, Matrix labels) {

        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features and labels must have the same number of rows.");
        }

        int n = features.getNumRows();

        if (n <= k) {
            if (labels.isCategorical(0)) {
                double index = labels.mostCommonValue(0);
                label = labels.getColumnAttributes(0).getValue((int) index);
            } else {
                labelValue = labels.columnMean(0);
            }
        } else if (labels.isHomogeneous(0)) {
            double index = labels.getRow(0).get(0);
            if (labels.isCategorical(0)) {
                label = labels.getColumnAttributes(0).getValue((int) index);
            } else {
                labelValue = index;
            }
        } else {
            handleSplit(features, labels);
        }
    }

    private void handleSplit(Matrix features, Matrix labels) {
        Matrix bestTrueFeatures = null, bestTrueLabels = null;
        Matrix bestFalseFeatures = null, bestFalseLabels = null;
        Split splitInfo = new Split();
        double minInfo = Double.MAX_VALUE;

        int n = features.getNumRows();

        for (int col = 0; col < features.getNumCols(); col++) {
            ColumnAttributes column = features.getColumnAttributes(col);

            if (features.isHomogeneous(col)) {
                double index = labels.getRow(0).get(0);
                if (labels.isCategorical(0)) {
                    label = labels.getColumnAttributes(0).getValue((int) index);
                } else {
                    labelValue = index;
                }
                return;
            }

            if (features.isCategorical(col)) {
                for (int val = 0; val < column.size(); val++) {
                    Matrix trueFeatures = new Matrix(features, true),
                            trueLabels = new Matrix(labels, true);
                    Matrix falseFeatures = new Matrix(features, true),
                            falseLabels = new Matrix(labels, true);

                    for (int row = 0; row < n; row++) {
                        double featureVal = features.getRow(row).get(col);
                        List<Double> currF = features.getRow(row);
                        List<Double> currL = labels.getRow(row);
                        if ((int) featureVal == val) {
                            trueFeatures.addRow(currF);
                            trueLabels.addRow(currL);
                        } else {
                            falseFeatures.addRow(currF);
                            falseLabels.addRow(currL);
                        }
                    }

                    if (trueFeatures.getNumRows() == 0 || falseFeatures.getNumRows() == 0) {
                        continue;
                    }

                    double trueEntropy, falseEntropy;

                    if (labels.isCategorical(0)) {
                        trueEntropy = getEntropy(trueFeatures, trueLabels, col);
                        falseEntropy = getEntropy(falseFeatures, falseLabels, col);
                    } else {
                        trueEntropy = trueLabels.variance(0);
                        falseEntropy = falseLabels.variance(0);
                    }

                    double info = (trueEntropy * (trueFeatures.getNumRows() / (double) n))
                            + (falseEntropy * (falseFeatures.getNumRows() / (double) n));

                    if (info < minInfo) {
                        bestTrueFeatures = trueFeatures;
                        bestTrueLabels = trueLabels;
                        bestFalseFeatures = falseFeatures;
                        bestFalseLabels = falseLabels;
                        splitInfo.columnName = column.getName();
                        splitInfo.columnValue = column.getValue(val);
                        splitInfo.columnType = column.getColumnType();
                        splitInfo.columnIndex = col;
                        minInfo = info;
                    }
                }

            } else {

                Matrix[] sample = sampleWithReplacement(features, labels, 8);

                for (int i = 0; i < sample[0].getNumRows(); i++) {
                    Matrix trueFeatures = new Matrix(features, true),
                            trueLabels = new Matrix(labels, true);
                    Matrix falseFeatures = new Matrix(features, true),
                            falseLabels = new Matrix(labels, true);

                    double divide = sample[0].getRow(i).get(col);

                    for (int row = 0; row < n; row++) {
                        List<Double> currF = features.getRow(row);
                        List<Double> currL = labels.getRow(row);
                        if (currF.get(col) < divide) {
                            trueFeatures.addRow(currF);
                            trueLabels.addRow(currL);
                        } else {
                            falseFeatures.addRow(currF);
                            falseLabels.addRow(currL);
                        }
                    }

                    if (trueFeatures.getNumRows() == 0 || falseFeatures.getNumRows() == 0) {
                        continue;
                    }

                    double trueEntropy, falseEntropy;

                    if (labels.isCategorical(0)) {
                        trueEntropy = getEntropy(trueFeatures, trueLabels, col);
                        falseEntropy = getEntropy(falseFeatures, falseLabels, col);
                    } else {
                        trueEntropy = trueLabels.variance(0);
                        falseEntropy = falseLabels.variance(0);
                    }

                    double info = (trueEntropy * (trueFeatures.getNumRows() / (double) n))
                            + (falseEntropy * (falseFeatures.getNumRows() / (double) n));

                    if (info < minInfo) {
                        bestTrueFeatures = trueFeatures;
                        bestTrueLabels = trueLabels;
                        bestFalseFeatures = falseFeatures;
                        bestFalseLabels = falseLabels;
                        splitInfo.columnType = column.getColumnType();
                        splitInfo.columnName = column.getName();
                        splitInfo.columnIndex = col;
                        splitInfo.columnMean = divide;
                        minInfo = info;
                    }
                }

            }
        }
        this.splitInfo = splitInfo;

        left = new DecisionTree(k);
        left.buildEntropyTree(bestTrueFeatures, bestTrueLabels);

        right = new DecisionTree(k);
        right.buildEntropyTree(bestFalseFeatures, bestFalseLabels);
    }

    public void buildRandomTree(Matrix features, Matrix labels) {

        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features and labels must have the same number of rows.");
        }

        int n = features.getNumRows();

        if (n <= k) {
            if (labels.isCategorical(0)) {
                double index = labels.mostCommonValue(0);
                label = labels.getColumnAttributes(0).getValue((int) index);
            } else {
                labelValue = labels.columnMean(0);
            }
        } else if (labels.isHomogeneous(0)) {
            double index = labels.getRow(0).get(0);
            if (labels.isCategorical(0)) {
                label = labels.getColumnAttributes(0).getValue((int) index);
            } else {
                labelValue = index;
            }
        } else {
            handleRandomSplits(features, labels);
        }

    }

    private void handleRandomSplits(Matrix features, Matrix labels) {
        Matrix trueFeatures = new Matrix(features, true),
                trueLabels = new Matrix(labels, true);
        Matrix falseFeatures = new Matrix(features, true),
                falseLabels = new Matrix(labels, true);
        Split splitInfo = new Split();

        int n = features.getNumRows();
        boolean foundSplit = false;

        while (!foundSplit) {
            int randCol = Rand.nextInt(features.getNumCols());
            ColumnAttributes column = features.getColumnAttributes(randCol);

            splitInfo.columnType = column.getColumnType();
            splitInfo.columnIndex = randCol;
            splitInfo.columnName = column.getName();

            if (features.isHomogeneous(randCol)) {
                double index = labels.getRow(0).get(0);
                if (labels.isCategorical(0)) {
                    label = labels.getColumnAttributes(0).getValue((int) index);
                } else {
                    labelValue = index;
                }
                return;
            }

            if (features.isCategorical(randCol)) {

                int randVal = Rand.nextInt(column.size());

                for (int row = 0; row < n; row++) {
                    double featureVal = features.getRow(row).get(randCol);
                    List<Double> currF = features.getRow(row);
                    List<Double> currL = labels.getRow(row);
                    if ((int) featureVal == randVal) {
                        trueFeatures.addRow(currF);
                        trueLabels.addRow(currL);
                    } else {
                        falseFeatures.addRow(currF);
                        falseLabels.addRow(currL);
                    }
                }

                splitInfo.columnValue = column.getValue(randVal);

            } else {

                Matrix[] sample = sampleWithReplacement(features, labels, 8);
                double sampleMean = sample[0].columnMean(randCol);

                for (int row = 0; row < n; row++) {
                    List<Double> currF = features.getRow(row);
                    List<Double> currL = labels.getRow(row);
                    if (currF.get(randCol) < sampleMean) {
                        trueFeatures.addRow(currF);
                        trueLabels.addRow(currL);
                    } else {
                        falseFeatures.addRow(currF);
                        falseLabels.addRow(currL);
                    }
                }

                splitInfo.columnMean = sampleMean;
            }

            if (trueFeatures.getNumRows() > 0 && falseFeatures.getNumRows() > 0) {
                foundSplit = true;
            } else {
                trueFeatures.clearData();
                trueLabels.clearData();
                falseFeatures.clearData();
                falseLabels.clearData();
            }
        }

        this.splitInfo = splitInfo;

        left = new DecisionTree(k);
        left.buildRandomTree(trueFeatures, trueLabels);

        right = new DecisionTree(k);
        right.buildRandomTree(falseFeatures, falseLabels);
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
            System.out.println("+" + parent + "->Is " + splitInfo.columnName + " == " + splitInfo.columnValue + "?");
            buffer.append("   |");
            left.prettyPrint(buffer, "Yes");
            buffer.setCharAt(buffer.length() - 1, ' ');
            right.prettyPrint(buffer, "No");
            buffer.delete(buffer.length() - 4, buffer.length());
        }
    }

    private static double getEntropy(Matrix features, Matrix labels, int col) {
        Map<Double, Integer> labelCount = new HashMap<Double, Integer>();
        for (int row = 0; row < features.getNumRows(); row++) {
            double label = labels.getRow(row).get(0);
            Integer count = labelCount.get(label);
            count = count == null ? 1 : count + 1;
            labelCount.put(label, count);
        }
        return getEntropy(labelCount, features.getNumRows());
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

}
