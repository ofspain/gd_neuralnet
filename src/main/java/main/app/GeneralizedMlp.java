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
import adaptive.nn.neuralnetwork.SynopticWeight;
import adaptive.nn.neuralnetwork.TanHFunction;
import java.util.Random;

/**
 *
 * @author femi
 */
public class GeneralizedMlp {

    private static final Random RANDOM_GENERATOR = new Random();
    private static final int EPOCH = 100000;
    private double eta = 0.1;
    private double momentum = 0.9;

    private final int n_input_neuron;
    private final int n_output_neuron;
    private final int[] n_hidden_neuron;
    private final SynopticWeight[][] input_hidden_connection;
    private final SynopticWeight[][] hidden_output_connection;
    private final SynopticWeight[][][] hidden_hidden_connection;

    private final Neuron[] input_neurons;
    private final Neuron[][] hidden_neurons;
    private final Neuron[] output_neurons;

    public GeneralizedMlp(int n_input, int[] n_hidden, int n_output) {
        n_input_neuron = n_input;
        n_hidden_neuron = n_hidden;
        n_output_neuron = n_output;

        int n_hidden_linked_input = n_hidden[0];
        int n_hidden_linked_output = n_hidden[n_hidden.length - 1];

        input_hidden_connection = new SynopticWeight[n_input_neuron][n_hidden_linked_input];
        hidden_output_connection = new SynopticWeight[n_output_neuron][n_hidden_linked_output];
        hidden_hidden_connection = new SynopticWeight[n_hidden.length - 1][][];

        output_neurons = new Neuron[n_output_neuron];
        input_neurons = new Neuron[n_input_neuron];
        hidden_neurons = new Neuron[n_hidden.length][];

        initializeNeurons(n_hidden);

        initializeWeight();
    }

    private void initializeNeurons(int[] n_hidden) {
        for (int i = 0; i < n_input_neuron; i++) {
            ActivationFunction af = new LinearFunction();
            Neuron n = new Neuron(af, false);
            input_neurons[i] = n;
        }

        for (int i = 0; i < n_hidden_neuron.length; i++) {
            Neuron[] hidden_i_layer = new Neuron[n_hidden[i]];
            for (int j = 0; j < hidden_i_layer.length; j++) {
                ActivationFunction af = new TanHFunction();
                Neuron n = new Neuron(af, true);
                hidden_i_layer[j] = n;
            }
            hidden_neurons[i] = hidden_i_layer;
        }

        for (int i = 0; i < n_output_neuron; i++) {
            ActivationFunction af = new LinearFunction();
            Neuron n = new Neuron(af, true);
            output_neurons[i] = n;
        }
    }

    private int n_last_hidden_neurons() {
        return n_hidden_neuron[n_hidden_neuron.length - 1];

    }

    private void initializeWeight() {

        for (int i = 0; i < n_input_neuron; i++) {
            for (int j = 0; j < n_hidden_neuron[0]; j++) {
                double rand_weight = RANDOM_GENERATOR.nextDouble();
                input_hidden_connection[i][j] = new SynopticWeight();
                input_hidden_connection[i][j].setWeight(rand_weight);
            }
        }

        for (int k = 0; k < n_output_neuron; k++) {
            for (int u = 0; u < n_last_hidden_neurons(); u++) {
                double rand_weight = RANDOM_GENERATOR.nextDouble();
                hidden_output_connection[k][u] = new SynopticWeight();
                hidden_output_connection[k][u].setWeight(rand_weight);
            }
        }
        //row is front, column is rare
        for (int r = 0; r < n_hidden_neuron.length - 1; r++) {
            int row = n_hidden_neuron[r];
            int column = n_hidden_neuron[r + 1];

            SynopticWeight[][] hidden_hidden = new SynopticWeight[row][column];

            for (int i = 0; i < row; i++) {
                for (int j = 0; j < column; j++) {
                    SynopticWeight s_w = new SynopticWeight();
                    s_w.setWeight(RANDOM_GENERATOR.nextDouble());
                    hidden_hidden[i][j] = s_w;
                }
            }
            hidden_hidden_connection[r] = hidden_hidden;
        }
    }
    
    private int number_hidden_neuron_in_first_layer(){
        return n_hidden_neuron[0];
    }
    
    private int number_hidden_neuron_last_layer(){
        int last_index = n_hidden_neuron.length - 1;
        return n_hidden_neuron[last_index];
    }
    
    private int number_hidden_neuron_next_to_last_layer(){
        int last_index = n_hidden_neuron.length - 1;
        return n_hidden_neuron[last_index-1];
    }
    
    

    public double[] predict(double[] inputs) {
        double[] result = new double[n_output_neuron];

        for (int i = 0; i < n_input_neuron; i++) {
            input_neurons[i].setInput(inputs[i]);
            input_neurons[i].applyActivation();
        }

        for (int i = 0; i < n_hidden_neuron[0]; i++) {
            double hidden_i_input = 0;
            for (int j = 0; j < n_input_neuron; j++) {
                double weight_input_hidden = input_hidden_connection[j][i].getWeight();
                double input_activated_output = input_neurons[j].getActivated_output();
                hidden_i_input += (weight_input_hidden * input_activated_output);
            }
            hidden_neurons[0][i].setInput(hidden_i_input);
            hidden_neurons[0][i].applyActivation();
        }

        
        for(int i=1; i<n_hidden_neuron.length; i++){
            Neuron[] to_activate = hidden_neurons[i];
            Neuron[] to_provide_input = hidden_neurons[i-1];
            
            for(int to_act=0; to_act<to_activate.length; to_act++ ){
                double input_to_use = 0.0;
                for(int to_pro=0; to_pro<to_provide_input.length; to_pro++){
                    double weight = hidden_hidden_connection[i-1][to_pro][to_act].getWeight();
                    double input_to_provide = to_provide_input[to_pro].getActivated_output();
                    input_to_use += (weight*input_to_provide);
                }
                hidden_neurons[i][to_act].setInput(input_to_use);
                hidden_neurons[i][to_act].applyActivation();
            }
        }

        for (int i = 0; i < n_output_neuron; i++) {
            double output_i_input = 0;
            for (int j = 0; j < n_last_hidden_neurons(); j++) {
                double weight_hidden_output = hidden_output_connection[i][j].getWeight();
                double hidden_activated_output = hidden_neurons[n_hidden_neuron.length - 1][j].getActivated_output();
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
                 //   System.out.println(predictions[j] + "|"+targets[j]);
                    errors[j] = targets[j] - predictions[j];
                    sum_error_square += (errors[j] * errors[j]);
                }

                backpropagateOutput(errors);
                backpropagateHidden();
                backpropagateInput();
            }
            double root_mean_error_square = (sum_error_square / pats.length);
            System.out.printf("ERROR IN " + i + " iteration is %.5f\n", root_mean_error_square);
        }
    }

    private void backpropagateHidden() {
        //update sigma
        for (int i = hidden_neurons.length - 1; i >= 1; i--) {
            Neuron[] column_neurons = hidden_neurons[i];
            Neuron[] row_neurons = hidden_neurons[i-1];
            
            for(int row=0; row<row_neurons.length;row++){
                ActivationFunction af = row_neurons[row].getAf();
                double input_row = row_neurons[row].getInput();
                double row_phi = af.differntiate(input_row);
                double sum_sigma_times_weight_to_column = 0.0;
                for(int column=0; column<column_neurons.length; column++){
                    double column_sigma = column_neurons[column].getLocal_induced_field_gradient__sigma();
                    double w_row_column = hidden_hidden_connection[i-1][row][column].getWeight();
                    sum_sigma_times_weight_to_column += (w_row_column * column_sigma);
                }
                
                double neuron_sigma = row_phi * sum_sigma_times_weight_to_column;
                hidden_neurons[i-1][row].setLocal_induced_field_gradient__sigma(neuron_sigma);
            }

        }

        for (int i = hidden_neurons.length - 1; i >= 1; i--) {
            Neuron[] column_neurons = hidden_neurons[i];
            Neuron[] row_neurons = hidden_neurons[i-1];
            // to update hidden_hidden_connection[i-1]
            
            for(int col=0; col<column_neurons.length; col++){
                Neuron to_bias_update = column_neurons[col];
                double col_sigma = to_bias_update.getLocal_induced_field_gradient__sigma();
                for(int row=0; row<row_neurons.length; row++){
                    Neuron neuron_to_link = row_neurons[row];
                    double y_row = neuron_to_link.getActivated_output();
                    
                    double new_delta = eta * y_row * col_sigma;
                    
                    SynopticWeight to_update = hidden_hidden_connection[i-1][row][col];
                    double old_weight = to_update.getWeight();
                    double old_delta = to_update.getLast_delta();
                    double new_weight = old_weight - new_delta + (momentum * old_delta);
                    
                    hidden_hidden_connection[i-1][row][col].setWeight(new_weight);
                    hidden_hidden_connection[i-1][row][col].setLast_delta(old_delta);
                    
                }
                /**
                 * Note the 1 in bias new_bias_delta is from 
                 * d(input_to_neuron)/d(weight_between_bias_and_neuron)
                 */
                double old_bias_weight = to_bias_update.getBias().getWeight();
                double old_bias_delta = to_bias_update.getBias().getLast_delta();
                double new_bias_delta = eta * 1 * col_sigma;
                double new_bias_weight = old_bias_weight - new_bias_delta + (momentum * old_bias_delta);
                
                hidden_neurons[i][col].getBias().setWeight(new_bias_weight);
                hidden_neurons[i][col].getBias().setLast_delta(new_bias_delta);
                
                
            }
        }
    }

    private void backpropagateOutput(double[] error) {

        //update sigma
        for (int i = 0; i < n_last_hidden_neurons(); i++) {
            Neuron hidden_last_layer_i = hidden_neurons[n_hidden_neuron.length - 1][i];
            double layer_last_i_input = hidden_last_layer_i.getInput();
            double layer_last_i_phi_prime = hidden_last_layer_i.getAf().differntiate(layer_last_i_input);
            double out_sum_sigma = 0.0;

            for (int j = 0; j < n_output_neuron; j++) {
                Neuron output_j = output_neurons[j];
                double input_j = output_j.getInput();
                double sigma_j = output_j.getAf().differntiate(input_j) * -1*error[j];
                double w_i_j = hidden_output_connection[j][i].getWeight();
                out_sum_sigma += (w_i_j * sigma_j);
            }
            double local_sigma_gradient = layer_last_i_phi_prime * out_sum_sigma;
            hidden_neurons[n_hidden_neuron.length - 1][i].setLocal_induced_field_gradient__sigma(local_sigma_gradient);
        }

        for (int i = 0; i < n_output_neuron; i++) {
            ActivationFunction af = output_neurons[i].getAf();
            double outputInput = output_neurons[i].getInput();
            double phi_prime = af.differntiate(outputInput);
            double error_2_correct = -1 * error[i];

            for (int j = 0; j < n_last_hidden_neurons(); j++) {
                double hidden_output = hidden_neurons[n_hidden_neuron.length - 1][j].getActivated_output();
                double new_delta = eta * phi_prime * hidden_output * error_2_correct;
                double old_weight = hidden_output_connection[i][j].getWeight();
                double old_delta = hidden_output_connection[i][j].getLast_delta();
                double new_weight = old_weight - new_delta + (momentum * old_delta);

                hidden_output_connection[i][j].setWeight(new_weight);
                hidden_output_connection[i][j].setLast_delta(new_delta);

                

            }
            
            double old_bias_weight = output_neurons[i].getBias().getWeight();
            double old_bias_delta = output_neurons[i].getBias().getLast_delta();
            double new_bias_delta = eta * phi_prime * 1 * error_2_correct;;
            double new_bias_weight = old_bias_weight - new_bias_delta + (momentum * old_bias_delta);
            output_neurons[i].getBias().setWeight(new_bias_weight);
            output_neurons[i].getBias().setLast_delta(new_bias_delta);
        }

    }

    private void backpropagateInput() {
        for (int i = 0; i < n_hidden_neuron[0]; i++) {
            ActivationFunction af = hidden_neurons[0][i].getAf();
            double hiddenInput = hidden_neurons[0][i].getInput();
            double hidden_phi_prime = af.differntiate(hiddenInput);
            double hidden_energies = 0.0;
            // System.out.println(hidden_hidden_connection[0].length);
            for (int j = 0; j < n_hidden_neuron[1]; j++) {

                double weight_output_hidden = hidden_hidden_connection[0][i][j].getWeight();
                hidden_energies += (hidden_neurons[1][j].getLocal_induced_field_gradient__sigma() * weight_output_hidden);

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

                double old_bias_weight = hidden_neurons[0][i].getBias().getWeight();
                double old_bias_delta = hidden_neurons[0][i].getBias().getLast_delta();
                double new_bias_weight = old_bias_weight - new_delta + (momentum * old_bias_delta);
                hidden_neurons[0][i].getBias().setWeight(new_bias_weight);
                hidden_neurons[0][i].getBias().setLast_delta(new_delta);
                //double new_std = getStandard_deviation() + std_delta + (0.01 * getLast_delta().get(STD_DEVIATION));
            }
        }
    }

}
