package diffevoltwovariables;

import java.util.*;

import java.util.stream.IntStream;

import structures.Point;

public class DBCVCalculator {

	private static double distances[][];


	private static double[][] calcPairwiseDistances(double[][] points) {
	    int numPoints = points.length;
	    double[][] pairwiseDistances = new double[numPoints][numPoints];

	    for (int i = 0; i < numPoints; i++) {
	        for (int j = 0; j < numPoints; j++) {
	            double distance = calcEuclideanDistance(points[i], points[j]);
	            pairwiseDistances[i][j] = distance;
	        }
	    }

	    return pairwiseDistances;
	}
	
	private static double calcEuclideanDistance(double[] x, double[] y) {

		double distValue = 0.0;

	
		long attributes = x.length;
		for (int attr = 0; attr < attributes; attr++) {
			distValue += Math.pow((x[attr] - y[attr]), 2);
		}

		return Math.sqrt(distValue);
	}


	private static class Edge {
		private int source;
		private int destination;
		private double weight;

		public Edge(int source, int destination, double weight) {
			this.source = source;
			this.destination = destination;
			this.weight = weight;
		}

		public int getSource() {
			return source;
		}

		public int getDestination() {
			return destination;
		}

		public double getWeight() {
			return weight;
		}
	}

	private static double squaredEuclideanDistance(double[] point1, double[] point2) {
		double sum = 0.0;
		for (int i = 0; i < point1.length; i++) {
			double diff = point1[i] - point2[i];
			sum += diff * diff;
		}
		return Math.sqrt(sum);
	}

	private static double coreDist(double[][] points, int indexOfPoint, int[] indicesOfNeighbors) {

		int nFeatures = points[indexOfPoint].length;

		double sum = 0.0;

		for (int otherPoint : indicesOfNeighbors) {
			double dist = distances[indexOfPoint][otherPoint];
			if (dist == 0.0)
				sum += 0.0; // Duplicate Objects in dataset
			else
				sum += Math.pow(1 / dist, nFeatures);
		}

		// double coreDist = Math.pow(numerator / (nNeighbors - 1), -1.0 / nFeatures);
		double coreDist = Math.pow((sum / (indicesOfNeighbors.length - 1)), -(1.0 / nFeatures));
		return coreDist;
	}

	private static double mutualReachabilityDist(double[][] points, int i, int j, int[] neighbors_i,
			int[] neighbors_j) {

		double coreDist_i = coreDist(points, i, neighbors_i);
		double coreDist_j = coreDist(points, j, neighbors_j);
		double distance = distances[i][j];
		double mutualReachability = Math.max(Math.max(coreDist_i, coreDist_j), distance);
		return mutualReachability;
	}

	private static double[][] mutualReachabilityDistGraph(double[][] points, int[] labels) {
		int nSamples = points.length;
		double[][] graph = new double[nSamples][nSamples];

		for (int i = 0; i < nSamples; i++) {
			for (int j = 0; j < nSamples; j++) {
				// double[][] neighbors_i = getLabelMembers(points, labels, labels[i]);
				// double[][] neighbors_j = getLabelMembers(points, labels, labels[j]);

				int[] neighbors_i = getLabelIndices(labels, labels[i]);
				int[] neighbors_j = getLabelIndices(labels, labels[j]);

				double mutualReachabilityDist = mutualReachabilityDist(points, i, j, neighbors_i, neighbors_j);
				graph[i][j] = mutualReachabilityDist;
				graph[j][i] = mutualReachabilityDist; // The graph is symmetric

			}
		}
		return graph;
	}


	private static double clusteringValidityIndex(List<Edge> mst, double[][] points, int[] labels) {
		int nSamples = points.length;
		double totalValidityIndex = 0.0;

		double[][] shortestPathWeights = new double[nSamples][nSamples];
		double[][] edgeWeights = new double[nSamples][nSamples];

		for (int i = 0; i < nSamples; i++) {
			for (int j = 0; j < nSamples; j++) {
				shortestPathWeights[i][j] = findShortestPathWeight(mst, i, j);
				edgeWeights[i][j] = findEdgeWeight(mst, i, j);
			}
		}

		
		
		for (int cluster : uniqueLabels(labels)) {
						
			int sizeofcluster = getLabelMembers(points, labels, cluster).length;
			double fraction = (((float) sizeofcluster) / nSamples);
			
			double clusterValidity = clusterValidityIndex(mst, points, labels, cluster, shortestPathWeights,
					edgeWeights);
			
			totalValidityIndex += (clusterValidity * fraction );
		}
		return totalValidityIndex;
	}

	private static double clusterValidityIndex(List<Edge> mst, double[][] points, int[] labels, int cluster,
			double[][] shortestPathWeights, double[][] edgeWeights) {
		int nSamples = points.length;
		double densitySeparation = Double.POSITIVE_INFINITY;
		double densitySparseness = Double.NEGATIVE_INFINITY;

		for (int i = 0; i < nSamples; i++) {
			if (labels[i] == cluster) {
				for (int j = 0; j < nSamples; j++) {
					if (labels[j] != cluster) {
						double minMutualReachabilityDist = Math.min(shortestPathWeights[i][j],
								shortestPathWeights[j][i]);
						densitySeparation = Math.min(densitySeparation, minMutualReachabilityDist);
					}
				}
			}
		}

		for (int i = 0; i < nSamples; i++) {
			if (labels[i] == cluster) {
				for (int j = 0; j < nSamples; j++) {
					if (labels[j] == cluster) {
						double mutualReachabilityDist = edgeWeights[i][j];
						densitySparseness = Math.max(densitySparseness, mutualReachabilityDist);
					}
				}
			}
		}

		double numerator = densitySeparation - densitySparseness;
		double denominator = Math.max(densitySeparation, densitySparseness);
		return numerator / denominator;
	}

	private static double findShortestPathWeight(List<Edge> mst, int i, int j) {
		Map<Integer, Double> distances = new HashMap<>();
		for (Edge edge : mst) {
			distances.put(edge.getSource(), Double.POSITIVE_INFINITY);
			distances.put(edge.getDestination(), Double.POSITIVE_INFINITY);
		}
		distances.put(i, 0.0);

		PriorityQueue<Integer> queue = new PriorityQueue<>(Comparator.comparingDouble(distances::get));
		queue.add(i);

		while (!queue.isEmpty()) {
			int u = queue.poll();
			if (u == j) {
				return distances.get(u);
			}

			for (Edge edge : mst) {
				if (edge.getSource() == u) {
					int v = edge.getDestination();
					double weight = edge.getWeight();
					if (distances.get(v) > distances.get(u) + weight) {
						distances.put(v, distances.get(u) + weight);
						queue.offer(v);
					}
				}
			}
		}

		// Destination vertex not reachable from source
		return Double.POSITIVE_INFINITY;
	}

	private static double findEdgeWeight(List<Edge> mst, int i, int j) {
		for (Edge edge : mst) {
			if (edge.getSource() == i && edge.getDestination() == j) {
				return edge.getWeight();
			}
		}
		return 0.0; // Return a sentinel value if the edge is not found
	}

	public static double calculateClusteringValidityIndex(double[][] points, int[] labels) {

		distances = calcPairwiseDistances(points);

		double[][] mrdGraph = mutualReachabilityDistGraph(points, labels);
		List<Edge> mst = minimumSpanningTree(mrdGraph);
		return clusteringValidityIndex(mst, points, labels);
	}

	private static List<Edge> minimumSpanningTree(double[][] distTree) {
		int n = distTree.length;
		List<Edge> edges = new ArrayList<>();

		// Populate the list of edges from the distTree
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				edges.add(new Edge(i, j, distTree[i][j]));
			}
		}

		// Sort the edges in non-decreasing order of weights
		edges.sort(Comparator.comparingDouble(Edge::getWeight));

		// Kruskal's algorithm to construct the minimum spanning tree
		List<Edge> mst = new ArrayList<>();
		UnionFind uf = new UnionFind(n);
		for (Edge edge : edges) {
			int u = edge.getSource();
			int v = edge.getDestination();
			if (!uf.isConnected(u, v)) {
				mst.add(edge);
				uf.union(u, v);
			}
		}

		return mst;
	}

	private static class UnionFind {
		private int[] parent;

		public UnionFind(int n) {
			parent = new int[n];
			for (int i = 0; i < n; i++) {
				parent[i] = i;
			}
		}

		public int find(int x) {
			if (x != parent[x]) {
				parent[x] = find(parent[x]);
			}
			return parent[x];
		}

		public void union(int x, int y) {
			int rootX = find(x);
			int rootY = find(y);
			if (rootX != rootY) {
				parent[rootX] = rootY;
			}
		}

		public boolean isConnected(int x, int y) {
			return find(x) == find(y);
		}
	}

	private static int[] uniqueLabels(int[] labels) {
		Set<Integer> labelSet = new HashSet<>();
		for (int label : labels) {
			labelSet.add(label);
		}
		int[] uniqueLabels = new int[labelSet.size()];
		int i = 0;
		for (int label : labelSet) {
			uniqueLabels[i++] = label;
		}
		return uniqueLabels;
	}

	private static int[] getLabelIndices(int[] labels, int cluster) {
		return IntStream.range(0, labels.length).filter(i -> labels[i] == cluster).toArray();
	}

	private static double[][] getLabelMembers(double[][] points, int[] labels, int cluster) {
		int count = countOccurrences(labels, cluster);
		double[][] members = new double[count][points[0].length];
		int index = 0;
		for (int i = 0; i < labels.length; i++) {
			if (labels[i] == cluster) {
				members[index++] = points[i];
			}
		}
		return members;
	}

	private static int countOccurrences(int[] array, int target) {
		int count = 0;
		for (int value : array) {
			if (value == target) {
				count++;
			}
		}
		return count;
	}

	public static void main(String[] args) {
		// Sample usage of the DBCV algorithm
		double[][] X = { { 0.1, 0.2 }, { 0.15, 0.25 }, { 0.12, 0.18 }, { 0.8, 0.85 }, { 0.85, 0.9 }, { 0.82, 0.87 } };

		int[] labels = { 0, 0, 0, 1, 1, 1 };

		int n = X.length;
		double[][] pairwiseDistances = new double[n][n];

		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++) {
				double sum = 0.0;
				for (int d = 0; d < X[i].length; d++) {
					double diff = X[i][d] - X[j][d];
					sum += diff * diff;
				}
				pairwiseDistances[i][j] = Math.sqrt(sum);
			}
		}

		distances = pairwiseDistances;
		double validityIndex = calculateClusteringValidityIndex(X, labels);
		System.out.println("Clustering Validity Index: " + validityIndex);
	}
}
