package structures;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

import org.jgrapht.Graph;
import org.jgrapht.alg.spanning.KruskalMinimumSpanningTree;
import org.jgrapht.graph.DefaultWeightedEdge;

public class Points extends ArrayList<Point> {
	private static final long serialVersionUID = 1L;
	private KruskalMinimumSpanningTree<Integer, DefaultWeightedEdge> mst;
	private Graph<Integer, DefaultWeightedEdge> mrdGraph;

	private double densitySparseness;
	private double validityIndex;

	public Points() {
	}

	public Points(Collection<Point> values) {
		super(values);
	}

	public Point getPointBySerialNumber(Integer serialNumber) {

		int ind = this.indexOf(serialNumber);

		if (ind != -1) {
			return this.get(ind);
		}

		return null;

	}

	@Override
	public int indexOf(Object o) {
		for (int i = 0; i < this.size(); i++) {
			Point p = this.get(i);
			if (p.serialNumber == (Integer) o) {
				return i;
			}
		}

		return -1;

	}

	public double getStandardDeviation(int paramIndex) {
		final double average = getAverage(paramIndex);
		double sum = 0.0;
		for (int i = 0; i < size(); i++) {
			final double diff = get(i).params[paramIndex] - average;
			sum += diff * diff;
		}
		return Math.sqrt(sum / size());
	}

	public double getAverage(int paramIndex) {
		double sum = 0.0;
		for (int i = 0; i < size(); i++) {
			sum += get(i).params[paramIndex];
		}
		return sum / size();
	}

	public int getDimenstion() {
		if (isEmpty()) {
			return 0;
		}
		return get(0).params.length;
	}

	public Clusters getPerfectClusters() {
		final Map<Integer, Points> clusterIdToCluster = new HashMap<Integer, Points>();
		for (final Point p : this) {
			Points cluster = clusterIdToCluster.get(p.trueClusterId);
			if (cluster == null) {
				cluster = new Points();
				cluster.add(p);
				clusterIdToCluster.put(p.trueClusterId, cluster);
			} else {
				cluster.add(p);
			}
		}
		return new Clusters(clusterIdToCluster.values());
	}

	public KruskalMinimumSpanningTree<Integer, DefaultWeightedEdge> getMst() {
		return mst;
	}

	public void setMst(KruskalMinimumSpanningTree<Integer, DefaultWeightedEdge> mst) {
		this.mst = mst;
	}

	public Graph<Integer, DefaultWeightedEdge> getMrdGraph() {
		return mrdGraph;
	}

	public void setMrdGraph(Graph<Integer, DefaultWeightedEdge> mrdGraph) {
		this.mrdGraph = mrdGraph;
	}

	public double getDensitySparseness() {
		return densitySparseness;
	}

	public void setDensitySparseness(double densitySparseness) {
		this.densitySparseness = densitySparseness;
	}

	public double getValidityIndex() {
		return validityIndex;
	}

	public void setValidityIndex(double validityIndex) {
		this.validityIndex = validityIndex;
	}
}
