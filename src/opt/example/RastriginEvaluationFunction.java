package opt.example;

import util.linalg.Vector;
import opt.EvaluationFunction;
import shared.Instance;

/**
 * A function that maximizes the negative of the Rastrigin function
 * This is equivalent to minimizing the actual Rastrigin
 * f(x) = -10 - x^2 +  10cos(2*pi*x)
 * Reference: http://www.sfu.ca/~ssurjano/rastr.html
 * By Laura Hamilton 9/19/2014
 */
public class RastriginEvaluationFunction implements EvaluationFunction {
    /**
     * @see opt.EvaluationFunction#value(opt.OptimizationData)
     */
    public double value(Instance d) {
        Vector data = d.getData();
        double val = 0;
        for (int i = 0; i < data.size(); i++) {
            val += data.get(i)*data.get(i) + 10*Math.cos(2*Math.PI*data.get(i));
        }
        return val;
    }
}
