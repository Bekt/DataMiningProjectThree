import ml.ARFFParser;
import ml.Filter;
import ml.Imputer;
import ml.Matrix;

public class Main {

    private static final int K = 1;

    public static void main(String[] args) throws Exception {

        Matrix matrix = ARFFParser.loadARFF("/Users/dev/workspace/DataMiningProjectThree/mushroom.arff");

        final int featuresStart = 0, featuresEnd = 22;
        final int labelsStart = 22, labelsEnd = 23;

        Matrix features = matrix.subMatrixCols(featuresStart, featuresEnd);
        Matrix labels = matrix.subMatrixCols(labelsStart, labelsEnd);

        DecisionTreeLearner learner = new DecisionTreeLearner(K);
        Imputer transformer = new Imputer();

        Filter filter = new Filter(learner, transformer, true);
        filter.train(features, labels);

        learner.printTree();
    }

}
