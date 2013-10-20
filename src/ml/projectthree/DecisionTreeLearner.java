package ml.projectthree;

import ml.MLException;
import ml.Matrix;
import ml.SupervisedLearner;

import java.util.ArrayList;
import java.util.List;

public class DecisionTreeLearner extends SupervisedLearner {

    DecisionTree tree;
    Matrix features, labels;
    int k;

    public DecisionTreeLearner(int k) {
        this.k = k;
    }

    @Override
    public void train(Matrix features, Matrix labels) {

        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features and labels must have the same number of rows.");
        }
        this.features = features;
        this.labels = labels;

        tree = new DecisionTree(k);
        tree.buildEntropyTree(features, labels);
    }

    @Override
    public List<Double> predict(List<Double> in) {
        DecisionTree node = tree;
        while (!node.isLeaf()) {
            int col = node.splitInfo.columnIndex;

            if (features.isCategorical(col)) {

                if (in.get(col).equals(features.getColumnAttributes(col).getIndex(node.splitInfo.columnValue))) {
                    node = node.left;
                } else {
                    node = node.right;
                }

            } else {
                if (in.get(col) < node.splitInfo.columnMean) {
                    node = node.left;
                } else {
                    node = node.right;
                }
            }
        }
        List<Double> out = new ArrayList<Double>();
        if (labels.isCategorical(0)) {
            out.add((double) labels.getColumnAttributes(0).getIndex(node.label));
        } else {
            out.add(node.labelValue);
        }
        return out;
    }

    public void printTree() {
        tree.prettyPrint(new StringBuilder(), "");
    }
}
