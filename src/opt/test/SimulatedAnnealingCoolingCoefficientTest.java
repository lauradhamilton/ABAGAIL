package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.ga.UniformCrossOver;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test of the knap sack problem
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class SimulatedAnnealingCoolingCoefficientTest {
    /** Random number generator */
    private static final Random random = new Random();
    /** The number of items */
    private static final int NUM_ITEMS = 40;
    /** The number of copies each */
    private static final int COPIES_EACH = 4;
    /** The maximum weight for a single element */
    private static final double MAX_WEIGHT = 50;
    /** The maximum volume for a single element */
    private static final double MAX_VOLUME = 50;
    /** The volume of the knapsack */
    private static final double KNAPSACK_VOLUME = 
         MAX_VOLUME * NUM_ITEMS * COPIES_EACH * .4;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        int[] copies = new int[NUM_ITEMS];
        Arrays.fill(copies, COPIES_EACH);
       double[] weights = {14.006873891334399,
0.9388306474803154,
26.119866173905713,
20.546411641966962,
12.456703411391896,
32.322455327605255,
38.750240535343956,
23.859794102127896,
45.011525134982286,
10.776070365757933,
46.065756634733006,
7.828572903429926,
24.743263676002634,
33.20916476805507,
16.51246534149665,
23.443925006858134,
18.57022530279454,
5.706896488446161,
23.44607697986756,
1.5545804205003566,
27.0302859936824,
21.097619402619628,
43.60173756385764,
49.44832347485482,
25.910156034801474,
27.91751206001118,
36.658173210220255,
40.881378221999206,
48.83228771437947,
35.49885544313467,
16.247757455771072,
25.53223124143824,
2.400993598957707,
12.408533226752189,
35.26405639169894,
46.35644830194322,
18.009317731328604,
47.96332151014204,
20.81843428091102,
15.819063866703608};
       double[] volumes={
44.57809711968351,
1.988378753951514,
47.593739208727456,
36.569659867427994,
1.5845284028427165,
41.7861748607473,
24.69594875244368,
43.03123587633294,
7.248980459072457,
23.327415667901835,
36.105898702916086,
24.87957120462097,
36.910177249731724,
27.30395021307583,
37.74427091808214,
21.681239410167937,
8.318371979533667,
16.88207551035857,
34.91767868192272,
9.456202413374893,
47.184521478105424,
35.65391669513947,
7.1158444301557155,
44.53433689634305,
49.16774307587011,
13.564617532166368,
38.36035512523829,
2.3636140632733618,
38.08282614908438,
49.310535335495906,
42.75495808871727,
28.422383043559908,
31.486561856652965,
7.283678338886252,
5.795560154240054,
38.749456539160306,
37.74109110751328,
0.7802313639065139,
19.468811616414722,
17.029576884574187};
       int[] ranges = new int[NUM_ITEMS];
        Arrays.fill(ranges, COPIES_EACH + 1);
        EvaluationFunction ef = new KnapsackEvaluationFunction(weights, volumes, KNAPSACK_VOLUME, copies);
System.out.println("Trial;Fitness;Running Time");
        for(double x=0; x<=1; x=x+.001){
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        SimulatedAnnealing sa = new SimulatedAnnealing(100, x, hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(sa, 500);
        fit = new FixedIterationTrainer(sa, 500);
        double sa_start = System.nanoTime();
        fit.train();
        double sa_end = System.nanoTime();
        double sa_trainingTime = sa_end - sa_start;
        sa_trainingTime /= Math.pow(10,9);
        

        /** Print out the values for each trial */

        System.out.println(x + ";" + ef.value(sa.getOptimal()) + ";" + sa_trainingTime);
    }
    }
}
