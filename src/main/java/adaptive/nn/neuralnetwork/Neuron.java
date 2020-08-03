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
public class Neuron {
    
    private final ActivationFunction af;
    private double input;
    private double activated_output;
    private final SynopticWeight bias;
    
    private double local_induced_field_gradient__sigma;
    
    public Neuron(ActivationFunction af,boolean biased){
        this.af = af;
        bias = new SynopticWeight();
        //
        bias.setWeight(biased ? new java.util.Random().nextDouble() : 0.0);
    }

    /**
     * @param input the input to set
     */
    public void setInput(double input) {
        this.input = input;
    }

    /**
     * @return the activated_output
     */
    public double getActivated_output() {
        return activated_output;
    }
    
    public void applyActivation(){
        activated_output = getAf().activate(getBias().getWeight()+getInput());
    }
    
    public double computeOutput(double connecting_weight){
        return activated_output * connecting_weight;
    }

    /**
     * @return the af
     */
    public ActivationFunction getAf() {
        return af;
    }

    /**
     * @return the input
     */
    public double getInput() {
        return input;
    }

    /**
     * @return the bias
     */
    public SynopticWeight getBias() {
        return bias;
    }

    /**
     * @return the local_induced_field_gradient__sigma
     */
    public double getLocal_induced_field_gradient__sigma() {
        return local_induced_field_gradient__sigma;
    }

    /**
     * @param local_induced_field_gradient__sigma the local_induced_field_gradient__sigma to set
     */
    public void setLocal_induced_field_gradient__sigma(double local_induced_field_gradient__sigma) {
        this.local_induced_field_gradient__sigma = local_induced_field_gradient__sigma;
    }
}
