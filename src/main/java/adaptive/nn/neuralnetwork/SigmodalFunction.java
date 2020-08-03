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
public class SigmodalFunction implements ActivationFunction{

    @Override
    public double activate(double input) {
        double result = 1;
        double exp = Math.exp(-1 * input);
        result +=exp;
        return 1 / result;
    }

    @Override
    public double differntiate(double input) {
        double exp = Math.exp(-1 * input);
        double denominator = Math.pow((1 + exp),2);
        return exp / denominator;
    }
    
}
