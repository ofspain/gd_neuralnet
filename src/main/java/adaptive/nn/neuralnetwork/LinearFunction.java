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
public class LinearFunction implements ActivationFunction{

    @Override
    public double activate(double input) {
        return input;
    }

    @Override
    public double differntiate(double input) {
        return 1;
    }
    
}
