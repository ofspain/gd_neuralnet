/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main.app;

import adaptive.nn.neuralnetwork.Pattern;

/**
 *
 * @author femi
 */
public class MainApp {

    public static void main(String[] args) {
        testMimo();
        //testSimpleSystem();
        // testmiso3();
    }

    private static void testMimo() {
        Pattern[] patterns = PatternBank.mimoSystem(2000);
        //int[] hidden_neuron = {20, 10};
       // GeneralizedMlp neuralNet = new GeneralizedMlp(4, hidden_neuron, 2);
        //neuralNet.trainNetwork(patterns);

         Network neuralNet = new Network(4,2);
        neuralNet.trainNetwork(patterns);
        // Pattern[] test_patterns = PatternBank.mimoSystem(200);
        //neuralNet.test_network(test_patterns);
    }

    private static void testSimpleSystem() {
        Pattern[] patterns = PatternBank.simpleSystem(1000);
        Network neuralNet = new Network(1, 1);
        neuralNet.trainNetwork(patterns);
    }

    private static void testmiso3() {
        Pattern[] patterns = PatternBank.misoSystemModel3(500);
        //int[] hidden_neuron = {1, 1};
        //GeneralizedMlp neuralNet = new GeneralizedMlp(2, hidden_neuron, 1);
        //neuralNet.trainNetwork(patterns);

        Network neuralNet = new Network(2, 1);
        neuralNet.trainNetwork(patterns);
    }
}
