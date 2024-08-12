package diffevoltwovariables;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Random;

public class NoiseAdder {

	public static final String datasetname = "wine_normalized";
    private static final String INPUT_FILE = "";  // Replace with your input file path
   

    public static void main(String[] args) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(INPUT_FILE));
            String line;

            // For each noise level
            for (double noiseLevel : new double[]{0.01, 0.05, 0.10, 0.20}) {
                FileWriter writer = new FileWriter(getOutputFileName(noiseLevel));

                while ((line = reader.readLine()) != null) {
                    String[] values = line.split(",");
                    for (int i = 0; i < values.length - 1; i++) {  // Excluding the last column (cluster label)
                        double value = Double.parseDouble(values[i]);
                        double noise = (2 * new Random().nextDouble() - 1) * noiseLevel;  // Random noise in [-noiseLevel, noiseLevel]
                        value += noise;
                        value = Math.max(0, Math.min(1, value));  // Ensuring values remain in [0, 1]
                        values[i] = String.valueOf(value);
                    }
                    writer.write(String.join(",", values) + "\n");
                }
                
                writer.close();
                reader.close();
                reader = new BufferedReader(new FileReader(INPUT_FILE));  // Resetting the reader
            }
            reader.close();

        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String getOutputFileName(double noiseLevel) {
        int percentage = (int) (noiseLevel * 100);
        return "";
    }
}
