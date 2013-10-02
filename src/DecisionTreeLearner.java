import ml.MLException;
import ml.Matrix;
import ml.SupervisedLearner;

import java.util.List;

public class DecisionTreeLearner extends SupervisedLearner {

    private DecisionTree tree;
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

        tree = new DecisionTree(features, labels, k);
    }

    @Override
    public List<Double> predict(List<Double> in) {
        throw new UnsupportedOperationException("Not implemented");
    }

    public void printTree() {
        tree.prettyPrint();
    }
}
