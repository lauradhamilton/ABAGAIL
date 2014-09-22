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

public class CountOnesGeneticTrial {
    /** The n value */
    private static final int N = 100;
    
    public static void main(String[] args) {
        System.out.println("Trial;Error;RunningTime");
        for(int x=1; x<=1000; x=x+1){
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

        int toMate = x/2;
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(x, toMate, 0, gap);
        FixedIterationTrainer fit = new FixedIterationTrainer(ga, 500);
        double ga_start = System.nanoTime();
        fit.train();
        double ga_end = System.nanoTime();
        double ga_trainingTime = ga_end - ga_start;
        ga_trainingTime /= Math.pow(10,9);

        /** Print out the values for each trial */

        System.out.println(x + ";" + ef.value(ga.getOptimal()) + ";" + ";" +  ga_trainingTime);
    }
    }
}
