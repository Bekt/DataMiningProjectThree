package ml.projectthree;

import ml.ARFFParser;
import ml.Filter;
import ml.Imputer;
import ml.Matrix;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class Main {

    private static final int K = 1;

    public static void main(String[] args) throws Exception {

        //Matrix matrix = ARFFParser.loadARFF("/Users/dev/workspace/DataMiningProjectThree/mushroom.arff");
        Matrix matrix = ARFFParser.loadARFF("/Users/dev/workspace/DataMiningProjectThree/housing.arff");
        //Matrix matrix = ARFFParser.loadARFF(args[0]);

        final int featuresStart = 0, featuresEnd = 12;
        final int labelsStart = 12, labelsEnd = 13;

        Matrix features = matrix.subMatrixCols(featuresStart, featuresEnd);
        Matrix labels = matrix.subMatrixCols(labelsStart, labelsEnd);

        DecisionTreeLearner learner = new DecisionTreeLearner(K);
        Imputer transformer = new Imputer();

        Filter filter = new Filter(learner, transformer, true);
        filter.train(features, labels);

        learner.printTree();
    }

}
