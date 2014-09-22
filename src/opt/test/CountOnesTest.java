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
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class CountOnesTest {
    /** The n value */
    private static final int N = 100;
    
    public static void main(String[] args) {
         System.out.println("Trial;RHF;SAF;GAF;MIMICF;RHT;SAT;GAT;MIMICT");
        for(int x=1; x<=100; x=x+1){
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new CountOnesEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new UniformCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 500);
        double rhc_start = System.nanoTime();
        fit.train();
        double rhc_end = System.nanoTime();
        double rhc_trainingTime = rhc_end - rhc_start;
        rhc_trainingTime /= Math.pow(10,9);
        
        SimulatedAnnealing sa = new SimulatedAnnealing(100, .95, hcp);
        fit = new FixedIterationTrainer(sa, 500);
        double sa_start = System.nanoTime();
        fit.train();
        double sa_end = System.nanoTime();
        double sa_trainingTime = sa_end - sa_start;
        sa_trainingTime /= Math.pow(10,9);
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(20, 20, 0, gap);
        fit = new FixedIterationTrainer(ga, 500);
        double ga_start = System.nanoTime();
        fit.train();
        double ga_end = System.nanoTime();
        double ga_trainingTime = ga_end - ga_start;
        ga_trainingTime /= Math.pow(10,9);
        
        MIMIC mimic = new MIMIC(50, 10, pop);
        fit = new FixedIterationTrainer(mimic, 500);
        double MIMIC_start = System.nanoTime();
        fit.train();
        double MIMIC_end = System.nanoTime();
        double MIMIC_trainingTime = MIMIC_end - MIMIC_start;
        MIMIC_trainingTime /= Math.pow(10,9);

        /** Print out the values for each trial */

        System.out.println(x + ";" + ef.value(rhc.getOptimal()) + ";" + ef.value(sa.getOptimal()) + ";" + ef.value(ga.getOptimal()) + ";" + ef.value(mimic.getOptimal())+ ";" + rhc_trainingTime + ";" + sa_trainingTime + ";" +  ga_trainingTime + ";" + MIMIC_trainingTime);
    }
    }
}
