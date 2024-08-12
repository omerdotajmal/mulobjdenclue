package diffevoltwovariables;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.math.BigDecimal;
import java.nio.file.FileSystems;
import java.nio.file.Path;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Calendar;
import java.util.Collections;
import java.util.Date;
import java.util.HashSet;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.PrimitiveIterator.OfDouble;
import java.util.Random;
import java.util.Set;
import java.util.StringTokenizer;
import java.util.stream.Collectors;
import java.util.stream.DoubleStream;

import org.jgrapht.Graph;
import org.jgrapht.alg.interfaces.SpanningTreeAlgorithm.SpanningTree;
import org.jgrapht.alg.spanning.KruskalMinimumSpanningTree;
import org.jgrapht.graph.DefaultUndirectedWeightedGraph;
import org.jgrapht.graph.DefaultWeightedEdge;

import com.opencsv.CSVWriter;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import algorithms.ClusteringAlgorithm;
import algorithms.Denclue;
import smile.validation.metric.AdjustedRandIndex;
import structures.Clusters;
import structures.Point;
import structures.Points;


public class DifferentialEvolutionComponent {
	private int diffEvol_dimension;
	private List<Bound> bounds;
	private List<Integer> integerIndices;
	private List<IntBound> integerBounds;
	private int populationSize;
	private double mutationFactor;
	private double crossoverRate;
	private int maxGenerations;
	private Points points;

	// Denclue + DBCV Code
	private List<List<Integer>> labelsList = new ArrayList<List<Integer>>();
	ArrayList<ArrayList<Integer>> clusterIndices = new ArrayList<ArrayList<Integer>>();
	int nClusters = 0;
	double ARI = 0.0;
	double dbcv = -1.0;
	double coverage = 0.0;
	static double distances[][];

	private String inputFile;
	private String outputFile;
	private String consoleOutputFile;
	static DecimalFormat df = new DecimalFormat("#.#######");
	static DecimalFormat df2 = new DecimalFormat("#.##");
	static DecimalFormat df_data = new DecimalFormat("#.##########");

	private StringBuffer consoleOutput = new StringBuffer();

	private List<List<Double>> dbcv_ari_pairs = new ArrayList<List<Double>>();
	// used for storing and writing individuals and objective function value at each
	// generation
	private List<OptimizationData> optimizationDataList = new ArrayList<>();

	// for DBCV Calculation , pointsForDBCV will be set in the beginning for
	// reducing complexity while
	// labelsforDBCV will be set after clustering from DENCLUE, in the method
	// runDenclue
	double[][] pointsForDBCV;
	int[] labelsforDBCV;

	public void setDEParameters(int dimension, List<Bound> bounds, List<Integer> integerIndices,
			List<IntBound> integerBounds, int populationSize, int DERun, int run, double mutationFactor,
			double crossoverRate, int maxGenerations, Points points, String outFile, String pairwiseDistFile) {
		this.diffEvol_dimension = dimension;
		this.bounds = bounds;
		this.integerIndices = integerIndices;
		this.integerBounds = integerBounds;
		this.populationSize = populationSize;
		this.mutationFactor = mutationFactor;
		this.crossoverRate = crossoverRate;
		this.maxGenerations = maxGenerations;
		this.points = points;


		this.outputFile = "";
	

		// calcPairwiseDistances(points);
		distances = new double[points.size()][points.size()];
		readPairwiseDistances(pairwiseDistFile);

	}

	public static double[][] convertPointsToDoubleArray(Points points) {
		int numPoints = points.size();
		int dimension = points.getDimenstion();

		double[][] pointsForDBCV = new double[numPoints][dimension];

		for (int i = 0; i < numPoints; i++) {
			Point point = points.get(i);
			double[] params = point.params;
			for (int j = 0; j < dimension; j++) {
				pointsForDBCV[i][j] = params[j];
			}
		}

		return pointsForDBCV;
	}

	private static void readPairwiseDistances(String inputFile) {

		try (BufferedReader reader = new BufferedReader(new FileReader(inputFile))) {
			String line;
			int i = 0;
			while ((line = reader.readLine()) != null) {
				String[] values = line.split(","); // Assuming the distances are separated by commas

				for (int j = 0; j < values.length; j++) {
					distances[i][j] = Double.parseDouble(values[j]);
				}

				i++;
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
	}

	// Method to initialize the population
	private List<List<Double>> initializePopulation() {
		List<List<Double>> population = new ArrayList<>();
		Random random = new Random();

		for (int i = 0; i < populationSize; i++) {
			List<Double> individual = new ArrayList<>();

			for (int j = 0; j < diffEvol_dimension; j++) {
				if (integerIndices.contains(j)) {
					int lowerBound = integerBounds.get(integerIndices.indexOf(j)).getLowerBound();
					int upperBound = integerBounds.get(integerIndices.indexOf(j)).getUpperBound();
					double value = lowerBound + (random.nextDouble() * (upperBound - lowerBound + 1));
					individual.add(value);
				} else {
					double lowerBound = bounds.get(j).getLowerBound();
					double upperBound = bounds.get(j).getUpperBound();
					double value = lowerBound + (random.nextDouble() * (upperBound - lowerBound));
					individual.add(value);
				}
			}

			population.add(individual);
		}

		return population;
	}

	// Method for mutation
	private List<List<Double>> mutatePopulation(List<List<Double>> population) {
		List<List<Double>> mutatedPopulation = new ArrayList<>();

		Random random = new Random();

		for (int i = 0; i < populationSize; i++) {
			List<Double> targetVector = population.get(i);

			// Select three distinct random individuals as base vectors: a, b, and c
			int a, b, c;
			do {
				a = random.nextInt(populationSize);
			} while (a == i);
			do {
				b = random.nextInt(populationSize);
			} while (b == i || b == a);
			do {
				c = random.nextInt(populationSize);
			} while (c == i || c == a || c == b);

			List<Double> mutantVector = new ArrayList<>();

			// Perform mutation for each variable in the population
			for (int j = 0; j < diffEvol_dimension; j++) {
				double aVal = population.get(a).get(j);
				double bVal = population.get(b).get(j);
				double cVal = population.get(c).get(j);

				// Calculate mutant vector using DE/rand/1/bin formula
				double mutant = aVal + mutationFactor * (bVal - cVal);

				if (integerIndices.contains(j)) {
					mutant = checkIntBounds(mutant, integerBounds.get(0));
				} else {
					mutant = checkBounds(mutant, bounds.get(0));
				}

				mutantVector.add(mutant);
			}

		
			mutatedPopulation.add(mutantVector);
		}

		return mutatedPopulation;
	}

	// Method for binomial crossover
	private List<List<Double>> crossoverPopulation(List<List<Double>> population, List<List<Double>> mutantPopulation) {
		List<List<Double>> offspringPopulation = new ArrayList<>();
		Random random = new Random();

		for (int i = 0; i < populationSize; i++) {
			List<Double> parent = population.get(i);
			List<Double> mutant = mutantPopulation.get(i);
			List<Double> offspring = new ArrayList<>();

			for (int j = 0; j < diffEvol_dimension; j++) {
				
				if (random.nextDouble() <= crossoverRate) {
					offspring.add(mutant.get(j)); // Perform crossover
				} else {
					offspring.add(parent.get(j));
				}
			}

			offspringPopulation.add(offspring);
		}

		return offspringPopulation;
	}

	private List<Double> calculateScalarizedValueForPopulation(List<List<Double>> population,
			List<List<Double>> popFitness) {

		int[] popParetoRank = new int[populationSize];
		double[] popWeights = new double[populationSize];
		double[] popScalarizedValue = new double[populationSize];

		double minDBCV = popFitness.stream().mapToDouble(solution -> solution.get(0)).min()
				.orElse(Double.POSITIVE_INFINITY);
		double maxDBCV = popFitness.stream().mapToDouble(solution -> solution.get(0)).max()
				.orElse(Double.NEGATIVE_INFINITY);

		for (int i = 0; i < populationSize; i++) {
			popParetoRank[i] = populationSize + 1;
		}

		for (int i = 0; i < populationSize; i++) {

			for (int j = 0; j < populationSize; j++) {
				if (i != j && solutionDominates(popFitness.get(i), popFitness.get(j))) {
					popParetoRank[i]--; // decrease the rank of the dominating solution
				}
			}
		}

		for (int i = 0; i < populationSize; i++) {
			// check if the solution is feasible, i.e., DBCV value not -1
			if (popFitness.get(i).get(0) != -1)
				popWeights[i] = 1.0 / popParetoRank[i];
			else
				popWeights[i] = 0.0;
		}
		double totalWeight = Arrays.stream(popWeights).sum();
		for (int i = 0; i < populationSize; i++) {

			if (popWeights[i] != 0.0) {
				popWeights[i] /= totalWeight;
				// solutions.get(i).setWeights(weights[i]);

				double normalizedDBCV = (popFitness.get(i).get(0) - minDBCV) / (maxDBCV - minDBCV);
				double normalizedCoverage = popFitness.get(i).get(1); // Coverage is already in [0, 1]

				// Calculate normalized scalarized value
				double scalarizedValue = weightedSum(normalizedDBCV, normalizedCoverage, popWeights[i]);
				popScalarizedValue[i] = scalarizedValue;

			} else {
				popScalarizedValue[i] = 0.0;
			}
		}

		List<Double> list = Arrays.stream(popScalarizedValue).boxed().collect(Collectors.toList());
		return list;

	}

	// Method for selection based on Scalarized
	private List<Object> selectPopulation(List<List<Double>> population, List<List<Double>> popFitness,
			List<List<Double>> trialVectors, List<List<Double>> trialFitness) {

		List<List<Double>> selectedPopulation = new ArrayList<>();
		List<List<Double>> selectedPopulationFitness = new ArrayList<>();
		// List<Double> selectedPopulationScalarizedValue = new ArrayList<>();

		List<Double> popScalarizedValue = calculateScalarizedValueForPopulation(population, popFitness);
		List<Double> trialScalarizedValue = calculateScalarizedValueForPopulation(trialVectors, trialFitness);

		for (int i = 0; i < populationSize; i++) {
			List<Double> popFit = popFitness.get(i);
			List<Double> trialFit = trialFitness.get(i);

			if (trialScalarizedValue.get(i) > popScalarizedValue.get(i)) { // Select trial vector
				selectedPopulation.add(trialVectors.get(i));
				selectedPopulationFitness.add(trialFit);
				// selectedPopulationScalarizedValue.add(trialScalarizedValue.get(i));

			} else {
				selectedPopulation.add(population.get(i)); // Select parent vector
				selectedPopulationFitness.add(popFit);
				// selectedPopulationScalarizedValue.add(popScalarizedValue.get(i));
			}

		}

		List<Double> selectedPopulationScalarizedValue = calculateScalarizedValueForPopulation(selectedPopulation,
				selectedPopulationFitness);

		ArrayList<Object> result = new ArrayList<>();

		result.add(selectedPopulation);
		result.add(selectedPopulationFitness);
		result.add(selectedPopulationScalarizedValue);

		return result;
	}

	// Method to evaluate the fitness of the population using DENCLUE clustering
	private List<List<Double>> evaluateFitness(List<List<Double>> population, int iter) {
		// Implement DENCLUE clustering and calculate DBCV value
		// Return a list of DBCV values for each candidate solution
		List<List<Double>> popFitness = new ArrayList<List<Double>>();

		long timeStart = Calendar.getInstance().getTimeInMillis();

		for (List<Double> ind : population) {
			float h = ind.get(0).floatValue();
			int minPts = ind.get(1).intValue();
			
			List<Double> objectiveValue = runDenclue(this.inputFile, h, minPts, iter);
			Double currARI = this.ARI;

			long timeEnd = Calendar.getInstance().getTimeInMillis();

			System.out.println("Time for 1 Denclue Execution:" + (timeEnd - timeStart));
			System.out.printf("Clusters = %d, h=%.7f, DBCV = %.7f, Coverage = %.2f, ARI = %.7f \n", nClusters, h,
					objectiveValue.get(0), objectiveValue.get(1), currARI);

			consoleOutput.append("\n");
			consoleOutput.append("Time for 1 Denclue Execution:" + (timeEnd - timeStart));
			consoleOutput.append("\n");
			consoleOutput.append("Clusters = " + nClusters + ", h = " + h + ", DBCV = " + objectiveValue.get(0)
					+ ", Coverage= " + objectiveValue.get(1) + ", ARI = " + currARI + "\n");

			popFitness.add(objectiveValue);

		}

		return popFitness;
	}

	private double checkBounds(double value, Bound bound) {
		// return Math.max(bound.getLowerBound(), Math.min(bound.getUpperBound(),
		// value));
		Random r = new Random();
		return ((r.nextDouble() * bound.getUpperBound() - bound.getLowerBound()) + bound.getLowerBound());
	}

	private double checkIntBounds(Double value, IntBound bound) {
		// return Math.max(bound.getLowerBound(), Math.min(bound.getUpperBound(),
		// value.intValue()));
		Random r = new Random();
		return ((r.nextDouble() * (bound.getUpperBound() - bound.getLowerBound() + 1)) + bound.getLowerBound());
	}

	// Method to perform the differential evolution optimization
	public void optimize() {
		// Initialize population
		List<List<Double>> population = initializePopulation();
		List<List<Double>> popFitness = evaluateFitness(population, -1);
		List<Double> populationScalarizedValue = calculateScalarizedValueForPopulation(population, popFitness);
		double avgScalarizedValue = 0.0;

		OptimizationData optimizationData = new OptimizationData();
		optimizationData.setGeneration(-1);
		optimizationData.setIndividualVectors(population);
		optimizationData.setObjectiveFunctionValues(popFitness);

		optimizationData.setScalarizedValue(populationScalarizedValue);
		// Add optimization data to the list
		optimizationDataList.add(optimizationData);

		List<Double> bestIndividual = new ArrayList<>();

		long timeStart = Calendar.getInstance().getTimeInMillis();

		// Perform optimization
		for (int generation = 0; generation < maxGenerations; generation++) {

			// Mutation
			List<List<Double>> mutantPopulation = mutatePopulation(population);

			// Crossover
			List<List<Double>> trialVectors = crossoverPopulation(population, mutantPopulation);

			// Evaluate fitness
			List<List<Double>> trialFitness = evaluateFitness(trialVectors, generation);

			// Selection
			List<Object> result = selectPopulation(population, popFitness, trialVectors, trialFitness);

			population = (List<List<Double>>) result.get(0);
			popFitness = (List<List<Double>>) result.get(1);
			populationScalarizedValue = (List<Double>) result.get(2);

			optimizationData = new OptimizationData();
			optimizationData.setGeneration(generation);
			optimizationData.setIndividualVectors(population);
			optimizationData.setObjectiveFunctionValues(popFitness);
			optimizationData.setScalarizedValue(populationScalarizedValue);
			// Add optimization data to the list
			optimizationDataList.add(optimizationData);

			int bestIndex = findBestIndividual(populationScalarizedValue);
			System.out.printf("\nIteration: %d f([%.7f, %d]) = [%.7f, %.2f]\n", generation,
					population.get(bestIndex).get(0).floatValue(), population.get(bestIndex).get(1).intValue(),
					popFitness.get(bestIndex).get(0), popFitness.get(bestIndex).get(1));

			avgScalarizedValue = populationScalarizedValue.stream().mapToDouble(Double::doubleValue).average()
					.orElse(Double.NaN);
			if (avgScalarizedValue >= 0.5)
				break;

		}

		
		long timeEnd = Calendar.getInstance().getTimeInMillis();
		long totalTime = timeEnd - timeStart;
		String dbcv_ari_file = outputFile + timeEnd + "_dbcv_ari.csv";
		String optimzation_data_file = outputFile + timeEnd + "_optimization_data.csv";

		writeDBCV_ARIPairs(dbcv_ari_file);
		writeOptimizationData(optimzation_data_file);

	}

	public static boolean solutionDominates(List<Double> solution1Fitness, List<Double> solution2Fitness) {

		if (solution1Fitness.get(0) > solution2Fitness.get(0)) {
			if (solution1Fitness.get(1) >= solution2Fitness.get(1)) {
				return true;
			}
		} else if (solution1Fitness.get(0) == solution2Fitness.get(0)) {
			if (solution1Fitness.get(1) > solution2Fitness.get(1)) {
				return true;
			}
		}

		return false;

	}

	public static double weightedSum(double dbcvValue, double coverageValue, double weight) {
		// Calculate scalarized value using weighted sum of objectives
		return weight * dbcvValue + (1 - weight) * coverageValue;
	}

	private int findBestIndividual(List<Double> populationScalarizedValue) {
		int bestIndex = 0;
		double bestFitness = populationScalarizedValue.get(bestIndex);

		// List<Double> currPopScalarizedValue =
		// calculateScalarizedValueForPopulation(population, fitness);

		for (int i = 1; i < populationSize; i++) {
			if (populationScalarizedValue.get(i) > bestFitness) {
				bestIndex = i;
				bestFitness = populationScalarizedValue.get(i);
			}
		}
		return bestIndex;
	}

	private ArrayList clustersToIndices(Clusters clusters) {

		ArrayList<ArrayList<Integer>> clusterIndices = new ArrayList<ArrayList<Integer>>();

		int pred_cluster = -1;
		for (Points cluster : clusters) {
			ArrayList<Integer> indicesOfSingleCluster = new ArrayList<Integer>();
			pred_cluster++;
			for (Point p : cluster) {
				indicesOfSingleCluster.add(p.serialNumber);
			}
			clusterIndices.add(indicesOfSingleCluster);

		}

		return clusterIndices;
	}

	private List clustersToList(Clusters clusters) {

		List<Integer> labels_true = new ArrayList<Integer>();
		List<Integer> labels_pred = new ArrayList<Integer>();

		int pred_cluster = -1;
		for (Points cluster : clusters) {
			pred_cluster++;
			for (Point p : cluster) {
				labels_true.add(p.trueClusterId);
				labels_pred.add(p.predClusterId);
			}
		}

		labelsList.add(labels_true);
		labelsList.add(labels_pred);

		return labelsList;
	}

	

	private double findMinDensitySeparationForCluster(Clusters clusters, int indexC, double[][] density_separation) {
		double minDSPC = Double.MAX_VALUE;

		for (int i = 0; i < clusters.size(); i++) {
			if (i == indexC)
				continue;

			if (density_separation[indexC][i] < minDSPC)
				minDSPC = density_separation[indexC][i];
		}

		return minDSPC;
	}

	/*
	 * the overall time complexity of the calcDensitySeparation method is O(n^2).
	 */
	private double calcDensitySeparation(Points cluster1, Points cluster2) {
		double minMRD = Double.MAX_VALUE;

		Set<Integer> internalVerticesCluster1 = getInternalVertices(cluster1);
		Set<Integer> internalVerticesCluster2 = getInternalVertices(cluster2);

		Iterator<Integer> cluster1Iterator = internalVerticesCluster1.iterator();
		while (cluster1Iterator.hasNext()) {
			Iterator<Integer> cluster2Iterator = internalVerticesCluster2.iterator();

			Point u = cluster1.getPointBySerialNumber(cluster1Iterator.next());

			while (cluster2Iterator.hasNext()) {
				Point v = cluster2.getPointBySerialNumber(cluster2Iterator.next());

				double dist_uv = calcSqEuclideanDistance(u, v);

				double mrd_uv = Double.max(Double.max(u.aptsCoreDist, v.aptsCoreDist), dist_uv);
				if (mrd_uv < minMRD)
					minMRD = mrd_uv;

			}

		}

		return minMRD;

	}

	// the overall time complexity of this method would be O(E + V), assuming that
	// checking the degree of a vertex and adding a vertex to the set takes constant
	// time.
	private Set<Integer> getInternalVertices(Points cluster) {
		Set<Integer> vertexSet = new HashSet<Integer>();

		Graph<Integer, DefaultWeightedEdge> graph = cluster.getMrdGraph();
		Graph<Integer, DefaultWeightedEdge> mstGraph = new DefaultUndirectedWeightedGraph<>(DefaultWeightedEdge.class);

		SpanningTree<DefaultWeightedEdge> mst_mrd = cluster.getMst().getSpanningTree();

		Iterator<DefaultWeightedEdge> edgesIter = mst_mrd.iterator();
		while (edgesIter.hasNext()) {

			DefaultWeightedEdge mstEdge = edgesIter.next();
			if (graph.containsEdge(mstEdge)) {
				mstGraph.addVertex(graph.getEdgeSource(mstEdge));
				mstGraph.addVertex(graph.getEdgeTarget(mstEdge));

				mstGraph.addEdge(graph.getEdgeSource(mstEdge), graph.getEdgeTarget(mstEdge), mstEdge);

			}

		}

		for (Integer v : mstGraph.vertexSet()) {
			if (mstGraph.degreeOf(v) > 1) {
				vertexSet.add(v);
			}

		}

		return vertexSet;

	}

	// So, the overall time complexity of the method is O(E), which is linear in the
	// number of edges in the MST.
	private void setDensitySparseness(Points cluster) {

		SpanningTree<DefaultWeightedEdge> mst_mrd = cluster.getMst().getSpanningTree();
		Graph<Integer, DefaultWeightedEdge> mrdGraph = cluster.getMrdGraph();

		Iterator<DefaultWeightedEdge> edgesIter = mst_mrd.iterator();

		double max_edge_wt = Double.MIN_VALUE;

		while (edgesIter.hasNext()) {

			DefaultWeightedEdge anEdge = edgesIter.next();
			if (mrdGraph.containsEdge(anEdge)) {
				Integer source = mrdGraph.getEdgeSource(anEdge);
				Integer target = mrdGraph.getEdgeTarget(anEdge);

				if (!(mrdGraph.degreeOf(source) == 1 || mrdGraph.degreeOf(target) == 1)) {
					double edgeWeight = mrdGraph.getEdgeWeight(anEdge);
					if (edgeWeight > max_edge_wt) {
						max_edge_wt = edgeWeight;
					}
				}
			}
		}

		cluster.setDensitySparseness(max_edge_wt);

	}

	private KruskalMinimumSpanningTree<Integer, DefaultWeightedEdge> getMST_MRDGraph(Points cluster,
			Graph<Integer, DefaultWeightedEdge> mrdGraph) {
		KruskalMinimumSpanningTree<Integer, DefaultWeightedEdge> mst = new KruskalMinimumSpanningTree<>(mrdGraph);

		return mst;

	}

	// O(N^2) N is the number of points in the graph
	private Graph<Integer, DefaultWeightedEdge> getMRDGraph(Points cluster, double[][] mrdMatrix) {

		Graph<Integer, DefaultWeightedEdge> mrdGraph = new DefaultUndirectedWeightedGraph<Integer, DefaultWeightedEdge>(
				DefaultWeightedEdge.class);

		for (Point p : cluster) {

			mrdGraph.addVertex(p.serialNumber);
		}

		for (Point p : cluster) {
			for (Point otherPoint : cluster) {

				if (p.serialNumber == otherPoint.serialNumber)
					continue;

				mrdGraph.addEdge(p.serialNumber, otherPoint.serialNumber);
				mrdGraph.setEdgeWeight(p.serialNumber, otherPoint.serialNumber,
						mrdMatrix[p.serialNumber][otherPoint.serialNumber]);

			}
		}

		return mrdGraph;

	}

	private void calcMRDInACluster(Point p, Points cluster, int dimension, double[][] mrdMatrix) {

		for (Point otherPoint : cluster) {
			mrdMatrix[p.serialNumber][otherPoint.serialNumber] = Math
					.max(Math.max(p.aptsCoreDist, otherPoint.aptsCoreDist), calcSqEuclideanDistance(p, otherPoint));
		}
	}

	private double calcAptsCoreDist(Point p, Points cluster, int dimension) {

		double sum = 0.0;

		for (Point otherPoint : cluster) {
			if (otherPoint.serialNumber == p.serialNumber)
				continue;

			double dist = distances[p.serialNumber][otherPoint.serialNumber];
			if (dist == 0.0)
				sum += 0.0; // Duplicate Objects in dataset
			else
				sum += Math.pow(1 / dist, dimension);
		}

		return Math.pow((sum / (cluster.size() - 1)), -(1.0 / dimension));

	}

	private static void calcPairwiseDistances(Points input) {
		distances = new double[input.size()][input.size()];
		try (NDManager manager = NDManager.newBaseManager()) {
			for (int i = 0; i < input.size(); i++) {
				for (int j = 0; j < input.size(); j++) {
					NDArray obj1 = manager.create(input.get(i).params);
					NDArray obj2 = manager.create(input.get(j).params);

					distances[input.get(i).serialNumber][input.get(j).serialNumber] = calcSqEuclideanDistance(obj1,
							obj2);

				}
			}
		}

	}

	private static double calcSqEuclideanDistance(Point x, Point y) {

		double distValue = 0.0;

		long attributes = x.params.length;

		for (int attr = 0; attr < attributes; attr++) {
			distValue += Math.pow((x.params[attr] - y.params[attr]), 2);
		}

		return distValue;
	}

	private static double calcSqEuclideanDistance(NDArray x, NDArray y) {

		double distValue = 0.0;

		// each object (x,y) is 1 row by n cols
		boolean dimOk = x.getShape().get(0) == y.getShape().get(0);
		if (!dimOk) {
			throw new IllegalArgumentException("Incorrect Dimensions");
		}

		long attributes = x.getShape().get(0);

		for (long attr = 0; attr < attributes; attr++) {
			distValue += Math.pow((x.getDouble(attr) - y.getDouble(attr)), 2);
		}

		return distValue;
	}

	private void writeConsoleOutput() {
		try {

			File file = new File(this.consoleOutputFile);
			// create FileWriter object with file as parameter
			FileWriter writer = new FileWriter(file);
			writer.write(consoleOutput.toString());
			writer.close();

		} catch (Exception e) {

		}

	}

	private void writeDBCV_ARIPairs(String filePath) {

		// first create file object for file placed at location
		// specified by filepath

		try {
			File file = new File(filePath);
			// create FileWriter object with file as parameter
			FileWriter outputfile = new FileWriter(file);

			// create CSVWriter object filewriter object as parameter
			CSVWriter writer = new CSVWriter(outputfile);

			ArrayList<String> headerList = new ArrayList<String>();

			headerList.add("Iteration/Gen");
			headerList.add("DBCV");
			headerList.add("Coverage");
			headerList.add("ARI");
			headerList.add("Sigma(h)");
			headerList.add("Minpts");
			headerList.add("Clusters");

			String[] header = new String[headerList.size()];
			headerList.toArray(header);
			writer.writeNext(header);

			String basicData[] = new String[] {
					" Writing DBCV ARI Pairs .., F = " + this.mutationFactor + ", CR = " + this.crossoverRate };
			writer.writeNext(basicData, false);

			for (List<Double> pair : this.dbcv_ari_pairs) {

				ArrayList<String> rowList = new ArrayList<String>();

				rowList.add(String.valueOf(pair.get(0)));
				rowList.add(df.format(pair.get(1)));
				rowList.add(df.format(pair.get(2)));
				rowList.add(df.format(pair.get(3)));
				rowList.add(df.format(pair.get(4)));
				rowList.add(String.valueOf(pair.get(5)));
				rowList.add(String.valueOf(pair.get(6)));

				String row[] = new String[rowList.size()];

				rowList.toArray(row);
				writer.writeNext(row, false);
			}

			writer.close();

		} catch (Exception e) {
			System.out.println(e.getMessage());
		}
	}

	private void writeOptimizationData(String filePath) {

		try (CSVWriter writer = new CSVWriter(new FileWriter(filePath))) {
			// Write the header row
			writer.writeNext(new String[] { "Generation", "h", "MinPts", "DBCV", "Coverage", "ScalarizedValue" });

			// Write the optimization data for each generation
			for (OptimizationData optimizationData : optimizationDataList) {

				for (int i = 0; i < this.populationSize; i++) {

					String[] rowData = new String[] { Integer.toString(optimizationData.getGeneration()),
							optimizationData.getIndividualVectors().get(i).get(0).toString(),
							optimizationData.getIndividualVectors().get(i).get(1).toString(),
							optimizationData.getObjectiveFunctionValues().get(i).get(0).toString(),
							optimizationData.getObjectiveFunctionValues().get(i).get(1).toString(),
							optimizationData.getScalarizedValue().get(i).toString() };

					writer.writeNext(rowData);

				}

			}

		} catch (IOException e) {
			e.printStackTrace();
		}

	}

	private void writeToOutputFile(String filePath, Clusters clusters, int[] labels_pred, int[] labels_true,
			int paramDimensions, int iter, float h, int minPts, double ARI, double currDBCV, double currCoverage) {

		if (paramDimensions <= 0) {
			throw new RuntimeException("Error in reading parameters for output file");
		}
		// first create file object for file placed at location
		// specified by filepath

		try {
			File file = new File(filePath);
			// create FileWriter object with file as parameter
			FileWriter outputfile = new FileWriter(file);

			// create CSVWriter object filewriter object as parameter
			CSVWriter writer = new CSVWriter(outputfile);

			String basicData[] = new String[] { " Iteration # " + iter, "Clusters = " + clusters.toString(), "h = " + h,
					" minPts=" + minPts, "DBCV =" + currDBCV, "Coverage =" + currCoverage, " ARI = " + ARI };
			writer.writeNext(basicData, false);

			ArrayList<String> headerList = new ArrayList<String>();
			for (int p = 0; p < paramDimensions; p++) {
				headerList.add("param" + p);
			}
			headerList.add("label_pred");
			headerList.add("label_true");

			String[] header = new String[headerList.size()];
			headerList.toArray(header);
			writer.writeNext(header);

			int i = -1;

			for (Points cluster : clusters) {
				++i;
				for (Point p : cluster) {

					ArrayList<String> rowList = new ArrayList<String>();

					for (int index = 0; index < paramDimensions; index++) {
						rowList.add(String.valueOf(p.params[index]));
					}
					rowList.add(String.valueOf(i));
					rowList.add(String.valueOf(p.trueClusterId));

					String row[] = new String[rowList.size()];

					// String[] row = {String.valueOf(p.params[0]), String.valueOf(p.params[1]),
					// String.valueOf(i), String.valueOf(p.clusterId)} ;
					rowList.toArray(row);
					writer.writeNext(row, false);
				}
			}

			writer.close();

		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	public List<Double> runDenclue(String fileName, float sigma, int minPts, int iter) {

		float sigmaFormatted = Float.parseFloat(df.format(sigma));

		double adjRIScore = 0.0;

		try {

			final Points input = points;
			int datasetDimension = points.get(0).params.length;

			final long startTime = System.currentTimeMillis();
			final long startFreeMemory = Runtime.getRuntime().freeMemory();

			ClusteringAlgorithm algorithm = new Denclue(input, sigmaFormatted, minPts);

			final Clusters clusters = algorithm.getClusters();

			final long stopFreeMemory = Runtime.getRuntime().freeMemory();
			final long stopTime = System.currentTimeMillis();
			final long elapsedTime = stopTime - startTime;
			final long usedMemory = startFreeMemory - stopFreeMemory;

			labelsList = new ArrayList<List<Integer>>();
			labelsList = clustersToList(clusters);

			this.nClusters = clusters.size();

			// These will only by used in the calculation of ARI, not for DBCV, as the
			// sequence is Cluster-wise here, and not
			// Point-wise as required by DBCV
			int[] labels_true = labelsList.get(0).stream().mapToInt(Integer::intValue).toArray();
			int[] labels_pred = labelsList.get(1).stream().mapToInt(Integer::intValue).toArray();

			labelsforDBCV = new int[labels_pred.length];
			pointsForDBCV = new double[labels_pred.length][labels_pred.length];

			// Setting predicted labels for DBCV Calculation
			int index = -1;
			for (Points cluster : clusters) {
				for (Point p : cluster) {
					++index;
					pointsForDBCV[index] = p.params;
					labelsforDBCV[index] = p.predClusterId;
				}
			}

			AdjustedRandIndex adjRIObj = new AdjustedRandIndex();
			adjRIScore = adjRIObj.score(labels_true, labels_pred);

			// System.out.print("ARI = " + adjRIScore);
			this.ARI = adjRIScore;

			this.clusterIndices = clustersToIndices(clusters);

			if (this.nClusters < 2) {
				this.dbcv = -1.0;
				this.coverage = 0.0;
				this.ARI = 0.0;
			} else {

				this.dbcv = DBCVCalculator.calculateClusteringValidityIndex(pointsForDBCV, labelsforDBCV);
				if (dbcv > 1.0) {
					System.out.print("DBCV greater than 1.0");
					System.exit(-1);
				}

				this.coverage = calculateCoverage(clusters, input);

			}

			long currTime = Calendar.getInstance().getTimeInMillis();
			String outputFilePath = this.outputFile + currTime + ".csv";

			// if (this.nClusters > 1) {

			List dbcv_ari = new ArrayList<Double>();
			dbcv_ari.add(iter);
			dbcv_ari.add(this.dbcv);
			dbcv_ari.add(this.coverage);
			dbcv_ari.add(this.ARI);
			dbcv_ari.add(sigma);
			dbcv_ari.add(minPts);
			dbcv_ari.add(clusters.toString());

			dbcv_ari_pairs.add(dbcv_ari);

			if (this.nClusters >= 2)
				writeToOutputFile(outputFilePath, clusters, labels_pred, labels_true, datasetDimension, iter, sigma,
						minPts, this.ARI, this.dbcv, this.coverage);
			// }

			// System.out.print("DBCV = " + this.dbcv);

		} catch (Exception e) {
			System.out.println(e.getMessage());

		}

		List<Double> objectiveValue = new ArrayList<>();
		objectiveValue.add(this.dbcv);
		objectiveValue.add(this.coverage);

		return objectiveValue;
	}

	private double calculateCoverage(Clusters clusters, Points input) {
		int clusteredPoints = 0;
		for (Points p : clusters) {
			clusteredPoints += p.size();
		}

		return (double) clusteredPoints / input.size();
	}

	private class OptimizationData {
		private int generation;

		private List<List<Double>> individualVectors;
		private List<List<Double>> objectiveFunctionValues;
		private List<Double> scalarizedValue;

		private OptimizationData() {

			this.generation = generation;
			this.individualVectors = individualVectors;
			this.objectiveFunctionValues = objectiveFunctionValues;
		}

		public int getGeneration() {
			return generation;
		}

		public void setGeneration(int generation) {
			this.generation = generation;
		}

		public List<List<Double>> getIndividualVectors() {
			return individualVectors;
		}

		public void setIndividualVectors(List<List<Double>> individualVectors) {
			this.individualVectors = individualVectors;
		}

		public List<List<Double>> getObjectiveFunctionValues() {
			return objectiveFunctionValues;
		}

		public void setObjectiveFunctionValues(List<List<Double>> objectiveFunctionValues) {
			this.objectiveFunctionValues = objectiveFunctionValues;
		}

		public List<Double> getScalarizedValue() {
			return scalarizedValue;
		}

		public void setScalarizedValue(List<Double> scalarizedValue) {
			this.scalarizedValue = scalarizedValue;
		}

	}

}
