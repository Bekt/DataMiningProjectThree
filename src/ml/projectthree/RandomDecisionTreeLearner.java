package ml.projectthree;

import ml.MLException;
import ml.Matrix;
import java.util.List;

public class RandomDecisionTreeLearner extends DecisionTreeLearner {

    public RandomDecisionTreeLearner(int k) {
        super(k);
    }

    @Override
    public void train(Matrix features, Matrix labels) {

        if (features.getNumRows() != labels.getNumRows()) {
            throw new MLException("Features and labels must have the same number of rows.");
        }
        this.features = features;
        this.labels = labels;

        tree = new DecisionTree(k);
        tree.buildRandomTree(features, labels);
    }

    @Override
    public List<Double> predict(List<Double> in) {
        return super.predict(in);
    }
}
