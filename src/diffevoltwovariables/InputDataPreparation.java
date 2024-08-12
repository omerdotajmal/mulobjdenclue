package diffevoltwovariables;

import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.StringTokenizer;

import structures.Point;
import structures.Points;

public class InputDataPreparation {
    private String inputFile;
    private Points points;

    public InputDataPreparation(String inputFile) {
        this.inputFile = inputFile;
        this.points = new Points();
    }

    public Points prepareData() {
        try {
            FileInputStream fin = new FileInputStream(inputFile);
            BufferedReader reader = new BufferedReader(new InputStreamReader(fin));
            String line;
            int serialNumber = 0;
            int dimension = 0;

            while ((line = reader.readLine()) != null) {
                StringTokenizer tokenizer = new StringTokenizer(line, ",");
                int numberOfParams = tokenizer.countTokens() - 1;
                dimension = numberOfParams;

                double[] params = new double[numberOfParams];
                for (int i = 0; i < numberOfParams; i++) {
                    String strVal = tokenizer.nextToken();
                    params[i] = Double.parseDouble(strVal);
                }
                int classId = Integer.parseInt(tokenizer.nextToken());
                Point point = new Point(params, classId, serialNumber);
                serialNumber++;

                points.add(point);
            }

            reader.close();
        } catch (Exception ex) {
            System.out.println("Error reading input file: " + ex.getMessage());
        }
        
        return points;
    }

    public Points getPoints() {
        return points;
    }
}
