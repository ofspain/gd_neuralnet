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
public class Pattern {

    private double[] input_values;
    private double[] desired_outputs;

    /**
     * @return the input_values
     */
    public double[] getInput_values() {
        return input_values;
    }

    /**
     * @param input_values the input_values to set
     */
    public void setInput_values(double[] input_values) {
        this.input_values = input_values;
    }

    /**
     * @return the desired_outputs
     */
    public double[] getDesired_outputs() {
        return desired_outputs;
    }

    /**
     * @param desired_outputs the desired_outputs to set
     */
    public void setDesired_outputs(double[] desired_outputs) {
        this.desired_outputs = desired_outputs;
    }

}
