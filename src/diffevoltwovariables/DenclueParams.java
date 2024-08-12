package diffevoltwovariables;

public class DenclueParams {
    private double bandwidth;
    private double convergenceThreshold;
    // Other parameters...

    public DenclueParams() {
        this.bandwidth = bandwidth;
        this.convergenceThreshold = convergenceThreshold;
        // Initialize other parameters...
    }
    
    public DenclueParams(double bandwidth, double convergenceThreshold) {
        this.bandwidth = bandwidth;
        this.convergenceThreshold = convergenceThreshold;
        // Initialize other parameters...
    }

    public double getBandwidth() {
        return bandwidth;
    }

    public double getConvergenceThreshold() {
        return convergenceThreshold;
    }

    // Getters and setters for other parameters...
}
