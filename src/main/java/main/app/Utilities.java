/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package main.app;

/**
 *
 * @author femi
 */
public class Utilities {

    public static void printArray(double[][] arrays) {
        for (double[] row : arrays) {
            System.out.println();
            double[] columns = row;
            printArray(columns);
        }
    }

    public static void printArray(double[] columns) {
        for (double column : columns) {
            System.out.print(column + " ");
        }
    }
}
