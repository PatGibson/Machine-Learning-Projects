import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Assignment 3, CMPE 452
 * @author Patrick Gibson 10148882
 *
 */
public class PCA {
	
	static ArrayList<Double> X1;	//Stores first column of .csv file
	static ArrayList<Double> X2;	//Stores 
	public static void main(String[] args) {
		getData();
		double meanX1 = getMean(1);	
		double meanX2 = getMean(23);
		double coVar = getCovariance(meanX1, meanX2);
		double varX1 = getVariance(meanX1, 1);
		double varX2 = getVariance(meanX2, 43);
		double[][] m = new double[2][2];	//2-D for covariance-variance matrix
		m[0][0] = varX1;
		m[0][1] = coVar;
		m[1][0] = coVar;
		m[1][1] = varX2;
		double posEigenValue = 0.5*((m[0][0] + m[1][1]) + Math.sqrt(4*m[0][1]*m[1][0] + 
				(m[0][0] - m[1][1])*(m[0][0] - m[1][1])));	//Eigen value positive root
		double negEigenValue = 0.5*((m[0][0] + m[1][1]) - Math.sqrt(4*m[0][1]*m[1][0] + 
				(m[0][0] - m[1][1])*(m[0][0] - m[1][1])));	//Eigen value negative root
		double winner = Math.max(posEigenValue, negEigenValue);
		double k = (winner - m[0][0])/m[0][1];	//solve for eigen vector (one element is 1)
		k = k /2;	//Divide by size of eigen vector. This value is used for weight value
		writeOutput(k);
	}
	
	/**
	 * 
	 * @param mean		mean of input
	 * @param choose	which input column
	 * @return
	 */
	public static double getVariance(double mean, int choose){
		double sum = 0;
		if(choose == 1){
			for(double x : X1)
				sum += (x - mean)*(x - mean);
		}
		else{
			for(double x : X2)
				sum += (x - mean)*(x - mean);
		}
		return sum/X1.size();
	}
	
	/**
	 * 
	 * @param meanX1	mean of first input
	 * @param meanX2	mean of second input
	 * @return	variance of inputs
	 */
	public static double getCovariance(double meanX1, double meanX2){
		double sum = 0;
		for(int i = 0; i < X1.size(); i++){
			sum += ((X1.get(i) - meanX1) * (X2.get(i) - meanX2));
		}
		return sum/X1.size();	
	}
	
	public static void getData(){
		String line;
		X1 = new ArrayList<Double>(50000);
		X2 = new ArrayList<Double>(50000);
		try {
			BufferedReader in = new BufferedReader(new FileReader("sound.csv"));
			while((line = in.readLine()) != null){
				String[] values = line.split(",");
				X1.add(Double.parseDouble(values[0]));
				X2.add(Double.parseDouble(values[1]));
			}
			in.close();
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}
	
	/**
	 * 
	 * @param choose	which input column
	 * @return	return mean of input
	 */
	public static double getMean(int choose){
		double sum = 0;
		if(choose == 1){
			for(double x : X1)
				sum += x;
		}
		else{
			for(double x : X2)
				sum += x;
		}
		return sum/X1.size();	
	}
	
	public static void writeOutput(double k){
		try {
			BufferedWriter out = new BufferedWriter(new FileWriter("Output.csv"));
			for(int i = 0; i < X1.size(); i++){
				double y = X1.get(i)*1 + X2.get(i)*k;
				out.write(Double.toString(y));
				out.write("\n");
			}
			out.close();
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
