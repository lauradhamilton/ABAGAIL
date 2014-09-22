package opt.test;

import java.util.Arrays;

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
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 100;
    /** The t value */
    private static final int T = N / 10;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 10000);
        double rhc_start = System.nanoTime();
        fit.train();
        double rhc_end = System.nanoTime();
        double rhc_trainingTime = rhc_end - rhc_start;
        rhc_trainingTime /= Math.pow(10,9);
        System.out.println("RHC: " + ef.value(rhc.getOptimal())+ " (" + rhc_trainingTime + " seconds)");
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainer(sa, 10000);
        double sa_start = System.nanoTime();
        fit.train();
        double sa_end = System.nanoTime();
        double sa_trainingTime = sa_end - sa_start;
        sa_trainingTime /= Math.pow(10,9);
        System.out.println("SA: " + ef.value(sa.getOptimal()) + " (" + sa_trainingTime + " seconds)");
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(500, 100, 10, gap);
        fit = new FixedIterationTrainer(ga, 10000);
        double ga_start = System.nanoTime();
        fit.train();
        double ga_end = System.nanoTime();
        double ga_trainingTime = ga_end - ga_start;
        ga_trainingTime /= Math.pow(10,9);
        System.out.println("GA: " + ef.value(ga.getOptimal())+ " (" + ga_trainingTime + " seconds)");
        
        MIMIC mimic = new MIMIC(500, 20, pop);
        fit = new FixedIterationTrainer(mimic, 10000);
        double MIMIC_start = System.nanoTime();
        fit.train();
        double MIMIC_end = System.nanoTime();
        double MIMIC_trainingTime = MIMIC_end - MIMIC_start;
        MIMIC_trainingTime /= Math.pow(10,9);
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal())+ " (" + MIMIC_trainingTime + " seconds)");
    }
}

