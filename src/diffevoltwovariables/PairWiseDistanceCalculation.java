package diffevoltwovariables;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import structures.Point;
import structures.Points;

public class PairWiseDistanceCalculation {

	public static void main(String[] args) {
		
		// Step 1: Prepare input data
		
		String[] datasetNames = new String[] {"iris_normalized", "processed_cleveland_normalized", "seeds_dataset_normalized", "wine_normalized",
				"iris_normalized_reduced", "processed_cleveland_normalized_reduced", "seeds_dataset_normalized_reduced", "wine_normalized_reduced" };
		
		for(String dataset:datasetNames)
		{
			//String dataset = "wine_normalized_reduced";
		
			String completeInputPath = "H:\\DropBox\\Dropbox\\OmarAjmal\\Publication\\Datasets\\" + dataset +"_outliers_20_percent.csv";
			
	        InputDataPreparation inputDataPreparation = new InputDataPreparation(completeInputPath);
	        Points points = inputDataPreparation.prepareData();
	        
			calcPairwiseDistances(points, completeInputPath.replace(".csv", "_pairwise_distancesEU.csv"));
		}

	}

	private static void calcPairwiseDistances(Points input, String outputFile) {
		
		try (PrintWriter writer = new PrintWriter(new FileWriter(outputFile))) {
			for (int i = 0; i < input.size(); i++) {
				for (int j = 0; j < input.size(); j++) {
					double distance = calcEuclideanDistance(input.get(i), input.get(j));

					// Save the distance to the CSV file
					writer.print(distance);
					if(j < input.size() - 1 )
						writer.print(","); // Use a comma (or any delimiter) to separate values
				}
				writer.println(); // Move to the next row in the CSV file
			}

		} catch (IOException e) {
			e.printStackTrace();
		}
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
	
	private static double calcEuclideanDistance(Point x, Point y) {

		double distValue = 0.0;

	
		long attributes = x.params.length;
		for (int attr = 0; attr < attributes; attr++) {
			distValue += Math.pow((x.params[attr] - y.params[attr]), 2);
		}

		return Math.sqrt(distValue);
	}
	
	private static double calcEuclideanDistance(NDArray x, NDArray y) {

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

		return Math.sqrt(distValue);
	}

}
