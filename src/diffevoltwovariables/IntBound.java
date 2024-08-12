package diffevoltwovariables;

public class IntBound {
    private int lowerBound;
    private int upperBound;

    public IntBound(int lowerBound, int upperBound) {
        this.lowerBound = lowerBound;
        this.upperBound = upperBound;
    }

    public int getLowerBound() {
        return lowerBound;
    }

    public int getUpperBound() {
        return upperBound;
    }
}