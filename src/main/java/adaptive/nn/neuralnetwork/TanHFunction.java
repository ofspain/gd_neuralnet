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
public class TanHFunction implements ActivationFunction {
    
    private final double a;
    private final double b;
    
    public TanHFunction(){
        this(1,1);
    }
    public TanHFunction(double a, double b){
        this.a = a;
        this.b = b;
    }

    @Override
    public double activate(double input) {
        double numerator = exponetial(b*input) - exponetial(-1*b*input);
        double denominator = exponetial(b*input) + exponetial(-1*b*input);
       // System.out.println(numerator +" : "+ denominator);
      
        return a * (numerator/denominator);
    }

    @Override
    public double differntiate(double input) {
        double denominator = exponetial(b*input) - exponetial(-1*b*input);
        double numerator = exponetial(b*input) + exponetial(-1*b*input);
       // System.out.println("diff "+denominator);
        return (a/b) * (numerator/denominator);
    }
    
    private double exponetial(double f){
        return Math.exp(f);
    }
    
}
