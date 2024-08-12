package diffevoltwovariables;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

import structures.Points;

public class Main {
	public static void main(String[] args) {

		String csvFile = "";
		List<List<Double>> data = new ArrayList<>();

		try (BufferedReader reader = new BufferedReader(new FileReader(csvFile))) {
			String line;
			boolean isFirstLine = true; // Skip the header line
			while ((line = reader.readLine()) != null) {
				if (isFirstLine) {
					isFirstLine = false;
					continue;
				}

				String[] values = line.split(",");

				// Extract the values for DE Run, F, and CR from the line
				int deRun = Integer.parseInt(values[0]);
				double f = Double.parseDouble(values[1]);
				double cr = Double.parseDouble(values[2]);

				// Create an ArrayList to store the current row data
				List<Double> rowData = new ArrayList<>();
				rowData.add((double) deRun); // Store DE Run as double
				rowData.add(f);
				rowData.add(cr);

				// Add the current row data to the main data ArrayList
				data.add(rowData);
			}
		} catch (IOException e) {
			e.printStackTrace();
		}

		

		String[] datasetNames = new String[] {""};
		
		int noisepct = 0;
		
		for (List<Double> row : data) {

			for (int run = 1; run <= 5; run++) {

				for (String ds : datasetNames) {

					
					
					String inputFile = "";
					
					// Step 1: Prepare input data
					InputDataPreparation inputDataPreparation = new InputDataPreparation(inputFile);
					Points points = inputDataPreparation.prepareData();

					// Step 2: Perform differential evolution optimization
					DifferentialEvolutionComponent deComponent = new DifferentialEvolutionComponent();
					int dimensions_DE = 2;
					int populationSize = 20;
					
					int DERun = row.get(0).intValue();
					double mutationFactor = row.get(1);
					double crossoverRate = row.get(2);
					
					
					int maxGenerations = 500;
					

					String pairWiseDistFile = "";

					
					deComponent.setDEParameters(dimensions_DE, createBounds(), createIntegerIndices(),
							createIntegerBounds(points.size()), populationSize, DERun, run, mutationFactor,
							crossoverRate, maxGenerations, points, ds, pairWiseDistFile);
					deComponent.optimize();

				}

			}

		}
	}

	private static List<Bound> createBounds() {
		List<Bound> bounds = new ArrayList<>();

		/*
		 * TODO: 
		 */
		bounds.add(new Bound(0.0000001, 1.0));
		
		return bounds;

	}

	private static List<Integer> createIntegerIndices() {
		List<Integer> intIndices = new ArrayList<>();
		intIndices.add(1);

		return intIndices;
	}

	private static List<IntBound> createIntegerBounds(int nPoints) {
		List<IntBound> bounds = new ArrayList<>();
		bounds.add(new IntBound(2, nPoints));

		return bounds;

	}
}
