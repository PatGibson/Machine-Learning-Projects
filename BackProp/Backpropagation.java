import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * This program is an artificial neural network using a backpropagation algorithm
 * to determine what the value of a handwritten digit is. It's currently unable to 
 * find optimal weight values, and therefore cannot determine the correct digit. 
 * @author Patrick Gibson
 *
 */
public class Backpropagation {
	static int[][] inputData;
	static int[][] testData;
	static double[][] inputWeights;
	static double[][] hiddenWeights;
	static double hiddenBias;
	static double outputBias; 
	final static double learningRate = 0.1; 

	
	/**
	 * This function is used to import the training data into an array. 
	 */
	public static void getData(){
		try {
			BufferedReader in = new BufferedReader(new FileReader("src/training.txt"));
			inputData = new int[3823][65];
			for(int i = 0; i < 3823; i++){
				String[] line = in.readLine().split(",");
				for(int j = 0; j < 65; j++){
					inputData[i][j] = Integer.parseInt(line[j]);
				}
			}
			in.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
	
	/**
	 * This function is used to import the test data into an array. 
	 */
	public static void getTestData(){
		try {
			BufferedReader in = new BufferedReader(new FileReader("src/testing.txt"));
			testData = new int[1797][65];
			for(int i = 0; i < 1797; i++){
				String[] line = in.readLine().split(",");
				for(int j = 0; j < 65; j++){
					testData[i][j] = Integer.parseInt(line[j]);
				}
			}
			in.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

	}
		
	/**
	 * This function generates random weights for starting values
	 */
	public static void randomHiddenWeights(){
		for(int i = 0; i < 4; i++){
			for(int j = 0; j < 12; j++) 
				hiddenWeights[i][j] = Math.random();
		}
		hiddenBias = Math.random();
	}
	
	/**
	 * This function generates random weights for starting values
	 */
	public static void randomInputWeights(){
		for(int i = 0; i < 12; i++){
			for(int j = 0; j < 64; j++) 
				inputWeights[i][j] = Math.random();
		}
		outputBias = Math.random();
	}
	
	/**
	 * This function is where the backpropagation algorithm is implemented. The while loop
	 * continues to iterate until the error is at an acceptable level. Inside the for loop, 
	 * every set of inputs is evaluated for every iteration of the while loop. The weights
	 * for the hidden and input layers are updated after every set of inputs.  
	 */
	public static void training(){
		double error = 1;
		int count = 0;
		double[] desiredOutput = null;			//Desired output based on inputs. 
		double[] hiddenOutput = new double[12]; //Output out of hidden layer.
		double[] finalOutput = new double[4];	//Output from final layer. 
		inputWeights = new double[12][64];
		hiddenWeights = new double[4][12];
		randomInputWeights();
		randomHiddenWeights();
		while(error > 0.05 && count < 10000){ //Continue to iterate until error is below 0.05, or after 10000 iterations 
			error  = 0;
			for(int i = 0; i < 3823; i ++){	//Loop through all possible inputs. 
				desiredOutput = new double[4];
				calculateHiddenOutput(hiddenOutput, i, inputData);		
				calculateFinalOutput(finalOutput, hiddenOutput, i);
				calculateDesiredOutput(desiredOutput, i, inputData);
				updateHiddenWeights(finalOutput, desiredOutput, hiddenOutput);
				updateInputWeights(finalOutput, desiredOutput, hiddenOutput, i);
				error += MSE(desiredOutput, finalOutput);
			}
			count++;
		}
	}
	
	/**
	 * Calculates outputs from hidden layer. 
	 */
	public static void calculateHiddenOutput(double[] output, int num, int[][] data){
		for(int i = 0; i < 12 ; i++){
			double sum = hiddenBias;
			for(int j = 0; j < 64; j++)
				sum += inputWeights[i][j] * data[num][j];
			output[i] = 1 / (1 + Math.exp(-sum));
		}		
	}
	
	/**
	 * Calculates final outputs from outer layer. 
	 */
	public static void calculateFinalOutput(double[] output, double[] input, int num){	
		for(int i = 0; i < output.length ; i++){
			double sum = outputBias;
			for(int j = 0; j < 12; j++){
				if(input[j] > 0.95)
					input[j] = 0.95;
				else if(input[j] < 0.05)
					input[j] = 0.05;
				sum += hiddenWeights[i][j] * input[j];
			}
			output[i] = 1 / (1 + Math.exp(-sum));
		}		
	}
	
	/**
	 * Function to set the desired output based on a set of inputs. The output is
	 * a binary number from the 4 output nodes. The values used are 0.05 and 0.95, instead of 0 
	 * and 1, since the Sigmoid function approaches 1 at infinity. 
	 * @param res
	 * @param num
	 */
	public static void calculateDesiredOutput(double[] res, int num, int[][] data){
		if(data[num][64] == 0){
			res[0] = 0.05; res[1]= 0.05; res[2] = 0.05; res[3] = 0.05;}
		else if(data[num][64] == 1){
			res[0] = 0.05; res[1] = 0.05; res[2] = 0.05; res[3] = 0.95;}
		else if(data[num][64] == 2){
			res[0] = 0.05; res[1]= 0.05; res[2] = 0.95; res[0] = 0.05;}
		else if(data[num][64] == 3){
			res[0] = 0.05; res[1] = 0.05; res[2] = 0.95; res[3] = 0.95;}
		else if(data[num][64] == 4){
			res[0] = 0.05; res[1] = 0.95; res[2] = 0.05; res[3] = 0.05;}
		else if(data[num][64] == 5){
			res[0] = 0.05; res[1] = 0.95; res[2] = 0.05; res[3] = 0.95;}
		else if(data[num][64] == 6){
			res[0] = 0.05; res[1] = 0.95; res[2] = 0.95; res[3] = 0.05;}
		else if(data[num][64] == 7){
			res[0] = 0.05; res[1] = 0.95; res[2] = 0.95; res[3] = 0.95;}
		else if(data[num][64] == 8){
			res[0] = 0.95; res[1] = 0.05; res[2] = 0.05; res[3] = 0.05;}
		else if(data[num][64] == 3){
			res[0] = 0.95; res[1] = 0.05; res[2] = 0.05; res[3] = 0.95;}
		}
	
	/**
	 * Updates weights going from hidden layer to output layer. 
	 * @param output final outputs
	 * @param dOutput Desired outputs
	 * @param input Output from hidden layer
	 */
	public static void updateHiddenWeights(double[] output, double[] dOutput, double[] input){
		for(int i = 0; i < 4; i++){
			if(output[i] > 0.95)
				output[i] = 0.95; //Change to 0.95 and 0.05 since outputs are these values instead of 1 and 0.
			else if(output[i] < 0.05)
				output[i] = 0.05;
			for(int j = 0; j < 12; j++){
				hiddenWeights[i][j] += learningRate * input[j] * (dOutput[i] - output[i]) * 
						output[i] * (1 - output[i]);
			}
		}
	}
	
	/**
	 * Updates weights going to hidden layer from inputs. 
	 * @param fOutput Final output
	 * @param dOutput Desired output
	 * @param hOutput Output from hidden layer
	 * @param num  Used to determine which row of input data to use
	 */
	public static void updateInputWeights(double[] fOutput, double[] dOutput, double[] hOutput, int num){
		for(int i = 0; i < 64; i++){
			for(int j = 0; j < 12; j++){
				for(int k = 0; k < 4; k++){
					inputWeights[j][i] += (dOutput[k] - fOutput[k])	* fOutput[k] * (1 - fOutput[k])
							* hiddenWeights[k][j] * hOutput[j] * (1 - hOutput[j]);
				}
			}
		}
	}
	
	public static double MSE(double[] dOutput, double[] fOutput){ //Calculate error for iteration. 
		double error = 0;
		for(int i = 0; i < 4; i++){	
			error += (fOutput[i] - dOutput[i]) * (fOutput[i] - dOutput[i]);
		}
		return error;
	}
	
	/**
	 * This function is used to try and predict the digits in the testing file. 
	 */
	public static void testing(){
		double[] desiredOutput = new double[4];			//Desired output based on inputs. 
		double[] hiddenOutput = new double[12]; //Output out of hidden layer.
		double[] finalOutput = new double[4];
		int correct = 0, wrong = 0;
		for(int i = 0; i < 1797; i++){
			calculateHiddenOutput(hiddenOutput, i, testData);
			calculateFinalOutput(finalOutput, hiddenOutput, i);
			calculateDesiredOutput(desiredOutput, i, testData);
			if(finalOutput[0] == desiredOutput[0] && finalOutput[1] == desiredOutput[1] && 
					finalOutput[2] == desiredOutput[2] && finalOutput[3] == desiredOutput[3])
				correct++;
			else
				wrong++;
		}
		System.out.println("Correct: " +correct);
		System.out.println("Wrong: " +wrong);
		System.out.println("Percentage correct: " +correct/wrong+ "%");
	}
	
	public static void main(String[] args){
		getData();
		training();	
		getTestData();
		testing();
	}
}
