/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main.app;

import adaptive.nn.neuralnetwork.Pattern;
import java.util.Random;

/**
 *
 * @author femi
 */
public class PatternBank {

    private static final Random RAND = new java.util.Random();

    public static Pattern[] mimoSystem(int smaple_data) {
        double y1_k = 0.0;
        double y2_k = 0.0;
        Pattern[] patterns = new Pattern[smaple_data];

        for (int i = 0; i < patterns.length; i++) {
            double u1_k = RAND.nextDouble() * 2 - 1;
            double u2_k = RAND.nextDouble() * 2 - 1;

            double y1_k_num = 0.5 * y1_k + 0.4 * u1_k + 0.6 * u2_k;
            double y1_k_denum = 1 + (y1_k * y1_k);
            double y1_k_plus_1 = y1_k_num / y1_k_denum;

            double y2_k_num = 0.5 * y2_k + 0.6 * u1_k + 0.4 * u2_k;
            double y2_k_denum = 1 + (y2_k * y2_k);
            double y2_k_plus_1 = y2_k_num / y2_k_denum;

            double[] target = new double[2];
            target[0] = y1_k_plus_1;
            target[1] = y2_k_plus_1;

            double[] input = new double[4];
            input[0] = u1_k;
            input[1] = u2_k;
            input[2] = y1_k;
            input[3] = y2_k;

            Pattern pattern = new Pattern();
            pattern.setDesired_outputs(target);
            pattern.setInput_values(input);

            patterns[i] = pattern;

            y1_k = y1_k_plus_1;
            y2_k = y2_k_plus_1;

        }

        return patterns;

    }

    public static Pattern[] simpleSystem(int smaple_data) {

        Pattern[] patterns = new Pattern[smaple_data];

        for (int i = 0; i < patterns.length; i++) {
            double u = i / 100;
            double v = (u * u * u) - (2 * u * u) + 7;

            double[] target = new double[1];
            target[0] = v;

            double[] input = new double[1];
            input[0] = u;

            Pattern pattern = new Pattern();
            pattern.setDesired_outputs(target);
            pattern.setInput_values(input);

            patterns[i] = pattern;

        }
        return patterns;
    }

    public static Pattern[] misoSystemModel3(int sample_data) {
        Pattern[] patterns = new Pattern[sample_data];

        double old_y = 0.0;
        for (int i = 0; i < sample_data; i++) {
            double result[] = evaluateModel_3(old_y);
            double u_k = result[0];
            double y_k_1 = result[1];
            double y_k = old_y;

            double[] target = new double[1];
            target[0] = y_k_1;
            
            double[] input = new double[2];
            input[0] = u_k; input[1] = y_k;
            
            Pattern pattern = new Pattern();
            pattern.setDesired_outputs(target);
            pattern.setInput_values(input);

            patterns[i] = pattern;
            old_y = y_k_1;
            
            //output.format("%.5f %.5f %.5f\n", u_k, y_k,y_k_1);
        }
        
        return patterns;
    }

    private static double[] evaluateModel_3(double old_y) {
        double input = new java.util.Random().nextDouble() * 2 - 1;
        double input_square = input * input * input;
        double denominator = (old_y * old_y) + 1;
        double term = old_y / denominator;
        double y_new = term + input_square;
        double result[] = new double[2];
        result[0] = input;
        result[1] = y_new;
        return result;

        /**
         * In general to get a value in the range [x,y) you should proceed thus
         * rand.nextDouble() * (x-y)+x
         */
    }
}
