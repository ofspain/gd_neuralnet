/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package adaptive.nn.neuralnetwork;

/**
 *
 * @author femi
 */
public class SynopticWeight {
    private double weight;
    private double last_delta;

    /**
     * @return the weight
     */
    public double getWeight() {
        return weight;
    }

    /**
     * @param weight the weight to set
     */
    public void setWeight(double weight) {
        if(weight > 1){
            weight = 1/weight;
        }
        this.weight = weight;
    }

    /**
     * @return the last_delta
     */
    public double getLast_delta() {
        return last_delta;
    }

    /**
     * @param last_delta the last_delta to set
     */
    public void setLast_delta(double last_delta) {
        this.last_delta = last_delta;
    }
}
