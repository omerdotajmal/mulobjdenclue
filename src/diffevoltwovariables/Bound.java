package diffevoltwovariables;

public class Bound {
    private double lowerBound;
    private double upperBound;

    public Bound(double lowerBound, double upperBound) {
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    public double getLowerBound() {
        return lowerBound;
    }

    public double getUpperBound() {
        return upperBound;
    }
}
