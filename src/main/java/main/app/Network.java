/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main.app;

import adaptive.nn.neuralnetwork.ActivationFunction;
import adaptive.nn.neuralnetwork.LinearFunction;
import adaptive.nn.neuralnetwork.Neuron;
import adaptive.nn.neuralnetwork.Pattern;
import adaptive.nn.neuralnetwork.SigmodalFunction;
import adaptive.nn.neuralnetwork.SynopticWeight;
import java.util.Random;

/**
 *
 * @author femi
 */
public class Network {

    private static final Random RANDOM_GENERATOR = new Random();
    private static final int EPOCH = 50000;
    private double eta = 0.1;
    private double momentum = 0.9;

    private static final int NUMBER_HIDDEN_NEURONS = 7;
    private final int n_input_neuron;
    private final int n_output_neuron;
    private final SynopticWeight[][] input_hidden_connection;
    private final SynopticWeight[][] hidden_output_connection;

    private final Neuron[] input_neurons;
    private final Neuron[] hidden_neurons;
    private final Neuron[] output_neurons;

    public Network(int n_input_variable, int n_output_variable) {
        n_input_neuron = n_input_variable;
        n_output_neuron = n_output_variable;
        input_hidden_connection = new SynopticWeight[n_input_neuron][NUMBER_HIDDEN_NEURONS];
        hidden_output_connection = new SynopticWeight[n_output_neuron][NUMBER_HIDDEN_NEURONS];

        input_neurons = new Neuron[n_input_neuron];
        hidden_neurons = new Neuron[NUMBER_HIDDEN_NEURONS];
        output_neurons = new Neuron[n_output_neuron];

        initializeWeight();
        initializeNeurons();
        System.out.println();

    }

    private void initializeWeight() {

        for (int i = 0; i < n_input_neuron; i++) {
            for (int j = 0; j < NUMBER_HIDDEN_NEURONS; j++) {
                double rand_weight = RANDOM_GENERATOR.nextDouble();
                input_hidden_connection[i][j] = new SynopticWeight();
                input_hidden_connection[i][j].setWeight(rand_weight);
            }
        }

        for (int k = 0; k < n_output_neuron; k++) {
            for (int u = 0; u < NUMBER_HIDDEN_NEURONS; u++) {
                double rand_weight = RANDOM_GENERATOR.nextDouble();
                hidden_output_connection[k][u] = new SynopticWeight();
                hidden_output_connection[k][u].setWeight(rand_weight);
            }
        }

    }

    private void initializeNeurons() {

        for (int i = 0; i < n_input_neuron; i++) {
            ActivationFunction af = new LinearFunction();
            Neuron n = new Neuron(af, false);
            input_neurons[i] = n;
        }

        for (int i = 0; i < NUMBER_HIDDEN_NEURONS; i++) {
            ActivationFunction af = new SigmodalFunction();
            Neuron n = new Neuron(af, true);
            hidden_neurons[i] = n;
        }

        for (int i = 0; i < n_output_neuron; i++) {
            ActivationFunction af = new LinearFunction();
            Neuron n = new Neuron(af, true);
            output_neurons[i] = n;
        }
    }

    /**
     * @param eta the eta to set
     */
    public void setEta(double eta) {
        this.eta = eta;
    }

    public double[] predict(double[] inputs) {
        double[] result = new double[n_output_neuron];

        for (int i = 0; i < n_input_neuron; i++) {
            input_neurons[i].setInput(inputs[i]);
            input_neurons[i].applyActivation();
        }

        for (int i = 0; i < Network.NUMBER_HIDDEN_NEURONS; i++) {
            double hidden_i_input = 0;
            for (int j = 0; j < n_input_neuron; j++) {
                double weight_input_hidden = input_hidden_connection[j][i].getWeight();
                double input_activated_output = input_neurons[j].getActivated_output();
                hidden_i_input += (weight_input_hidden * input_activated_output);
            }
            hidden_neurons[i].setInput(hidden_i_input);
            hidden_neurons[i].applyActivation();
        }

        for (int i = 0; i < n_output_neuron; i++) {
            double output_i_input = 0;
            for (int j = 0; j < Network.NUMBER_HIDDEN_NEURONS; j++) {
                double weight_hidden_output = hidden_output_connection[i][j].getWeight();
                double hidden_activated_output = hidden_neurons[j].getActivated_output();
                output_i_input += (weight_hidden_output * hidden_activated_output);
            }
            output_neurons[i].setInput(output_i_input);
            output_neurons[i].applyActivation();
        }

        for (int i = 0; i < n_output_neuron; i++) {
            result[i] = output_neurons[i].getActivated_output();
        }
        return result;
    }

    public void trainNetwork(Pattern[] pats) {
        for (int i = 0; i < EPOCH; i++) {
            double sum_error_square = 0.0;
            for (Pattern pat : pats) {
                double[] inputs = pat.getInput_values();
                double targets[] = pat.getDesired_outputs();
                double[] predictions = this.predict(inputs);
                double[] errors = new double[n_output_neuron];
                for (int j = 0; j < errors.length; j++) {
                    errors[j] = targets[j] - predictions[j];
                    sum_error_square += (errors[j] * errors[j]);
                }

                backpropagateOutput(errors);
                backpropagateHidden(errors);
            }
            double root_mean_error_square = (sum_error_square / pats.length);
            System.out.printf("ERROR IN " + i + " iteration is %.5f\n", root_mean_error_square);
        }
    }

    public void test_network(Pattern[] patterns) {
        double sum_error_square = 0.0;
        for (Pattern pat : patterns) {
            double[] target = pat.getDesired_outputs();
            double[] input = pat.getInput_values();
            double prediction[] = predict(input);
            

            for (int i = 0; i < target.length; i++) {
                double error = target[i] - prediction[i];
                System.out.println(target[i] + " | " + prediction[i] + " | " + error);

                sum_error_square += (error * error);
            }
        }
        double root_mean_error_square = (sum_error_square / patterns.length);
        System.out.println("ERROR IN  test "+ root_mean_error_square);
    }

    private void backpropagateOutput(double[] error) {

        for (int i = 0; i < n_output_neuron; i++) {
            ActivationFunction af = output_neurons[i].getAf();
            double outputInput = output_neurons[i].getInput();
            double phi_prime = af.differntiate(outputInput);
            double error_2_correct = -1 * error[i];
            for (int j = 0; j < Network.NUMBER_HIDDEN_NEURONS; j++) {
                double hidden_output = hidden_neurons[j].getActivated_output();
                double new_delta = eta * phi_prime * hidden_output * error_2_correct;
                double old_weight = hidden_output_connection[i][j].getWeight();
                double old_delta = hidden_output_connection[i][j].getLast_delta();
                double new_weight = old_weight - new_delta + (momentum * old_delta);

                hidden_output_connection[i][j].setWeight(new_weight);
                hidden_output_connection[i][j].setLast_delta(new_delta);

                //double new_std = getStandard_deviation() + std_delta + (0.01 * getLast_delta().get(STD_DEVIATION));
            }

            double old_bias_weight = output_neurons[i].getBias().getWeight();
            double old_bias_delta = output_neurons[i].getBias().getLast_delta();
            double new_bias_delta = eta * phi_prime * 1 * error_2_correct;;
            double new_bias_weight = old_bias_weight - new_bias_delta + (momentum * old_bias_delta);
            output_neurons[i].getBias().setWeight(new_bias_weight);
            output_neurons[i].getBias().setLast_delta(new_bias_delta);

        }
    }

    private void backpropagateHidden(double[] error) {
        for (int i = 0; i < NUMBER_HIDDEN_NEURONS; i++) {
            ActivationFunction af = hidden_neurons[i].getAf();
            double hiddenInput = hidden_neurons[i].getInput();
            double hidden_phi_prime = af.differntiate(hiddenInput);
            double hidden_energies = 0.0;
            for (int j = 0; j < n_output_neuron; j++) {
                ActivationFunction output_af = output_neurons[j].getAf();
                double outputInput = output_neurons[j].getInput();
                double error_2_correct = -1 * error[j];

                double output_phi_prime = output_af.differntiate(outputInput);
                double output_sigma = error_2_correct * output_phi_prime;
                double weight_output_hidden = hidden_output_connection[j][i].getWeight();
                hidden_energies += (output_sigma * weight_output_hidden);

            }
            double hidden_local_induced_field = hidden_phi_prime * hidden_energies;

            for (int k = 0; k < this.n_input_neuron; k++) {
                double input_output = input_neurons[k].getInput();
                double new_delta = eta * hidden_local_induced_field * input_output;
                double old_weight = input_hidden_connection[k][i].getWeight();
                double last_delta = input_hidden_connection[k][i].getLast_delta();
                double new_weight = old_weight - new_delta + (momentum * last_delta);
                input_hidden_connection[k][i].setWeight(new_weight);
                input_hidden_connection[k][i].setLast_delta(new_delta);

                //double new_std = getStandard_deviation() + std_delta + (0.01 * getLast_delta().get(STD_DEVIATION));
            }

            double old_bias_weight = hidden_neurons[i].getBias().getWeight();
            double old_bias_delta = hidden_neurons[i].getBias().getLast_delta();
            double new_bias_delta = eta * hidden_local_induced_field * 1;
            double new_bias_weight = old_bias_weight - new_bias_delta + (momentum * old_bias_delta);
            hidden_neurons[i].getBias().setWeight(new_bias_weight);
            hidden_neurons[i].getBias().setLast_delta(new_bias_delta);

        }

    }

}
