import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class PerceptronFL {
	private double[] w;							//Weight values
	private String[] values;					//Holds lines of input data
	private String[] testValues;				//Holds lines of output data
	private double[] x1 = new double[120];		//First input, Sepal length
	private double[] x2 = new double[120];		//Second input, Sepal width
	private double[] x3 = new double[120];		//Third input, Petal length
	private double[] x4 = new double[120];		//Fourth input, Petal width
	private String[] name = new String[120];	//Name of flower species
	final double learningRate = 0.1;			
	int theta = 0;
	String[] percepTest;
	String[] errorTest;
	
	/**
	 * This constructor is used to import the training and testing data 
	 * into appropriate arrays.
	 * @throws IOException
	 */
	PerceptronFL() throws IOException{
		w = new double[5];
		values = new String[120];
		BufferedReader in = new BufferedReader(new FileReader("src/train.txt"));
	    for(int i = 0; i < 120; i++){
	    	values[i] = in.readLine(); //inputs every line of data into array
	    }
	    String[] line;
	    for(int i = 0; i < values.length; i++){ //Splits data into 5 seperate inputs
	    	line = values[i].split(",");
	    	x1[i] = Double.parseDouble(line[0]); 
	    	x2[i] = Double.parseDouble(line[1]); 
	    	x3[i] = Double.parseDouble(line[2]); 
	    	x4[i] = Double.parseDouble(line[3]); 
	    	name[i] = line[4]; //class
	    }
	    in.close();
	    
	    testValues = new String[30];
	    BufferedReader in2 = new BufferedReader(new FileReader("src/test.txt"));
	    for(int i = 0; i < 30; i++){
	    	testValues[i] = in2.readLine(); //import testing data
	    }
	    in2.close();
	}
	
	/**
	 * This function is used to adjust the weights to learn the separation between
	 * Iris-setosa and not Iris-setosa. A perceptron algorithm is used to adjust the 
	 * weights.
	 */
	public void trainPerceptron(){
		int count = 0, e = 0; //Counts number of iterations
		int desiredOutput, output; 	//Calculated output and expected output
		double lError;	//error values
		randomWeights(); //Set weights to initial random values

		do{ 
			count++;
			e = 0;
			for(int i = 0; i < x1.length; i++){ //Test each data point
				output = calculateOutput(x1[i], x2[i], x3[i], x4[i]); //determine calculated output
				desiredOutput = calculateDesiredOutput(name[i]); //determine real output
				lError = desiredOutput - output;
				if(lError == 0) //If no error has occurred 
					e++;
				//update weight values
				w[0] += learningRate * lError * x1[i];
				w[1] += learningRate * lError * x2[i];
				w[2] += learningRate * lError * x3[i];
				w[3] += learningRate * lError * x4[i];
				w[4] += learningRate * lError;				
			}
		}while(x1.length - e != 0); //iterate until no errors
       System.out.println("\nNumber of iterations for perceptron: " +count); 
	}
	
	
	/**
	 * This function is used to predict if a set of inputs belong to Iris-setosa 
	 * or not based on the weights set in the trainPerceptron algorithm.
	 * @throws IOException
	 */
	public void testPerceptron() throws IOException{
		//Get test data 
		double testx1[] = new double[30];
		double testx2[] = new double[30];
		double testx3[] = new double[30];
		double testx4[] = new double[30];
		String[] testName = new String[30];
	    String[] line;
	    //Set arrays for test data points
	    for(int i = 0; i < testValues.length; i++){
	    	line = testValues[i].split(",");
	    	testx1[i] = Double.parseDouble(line[0]);
	    	testx2[i] = Double.parseDouble(line[1]);
	    	testx3[i] = Double.parseDouble(line[2]);
	    	testx4[i] = Double.parseDouble(line[3]);
	    	testName[i] = line[4];
	    }
	    percepTest = new String[30];
	    for(int i = 0; i < testx1.length; i++){ //Test every data set
	    	int output = calculateOutput(testx1[i], testx2[i], testx3[i], testx4[i]);
	    	if(output == 1) //Output of 1 indicates Setosa
	    		percepTest[i] = "Iris-Setosa";
	    	else
	    		percepTest[i] = "Not Setosa";
	    }
	    
	}

	
	/**
	 * This function is used to determine if a set of inputs belongs to Iris-versicolor 
	 * or to Iris-viriginica. This algorithm uses error correction learning to set
	 * the desired weight values. 
	 */
	public void trainErrorCorrection(){
		int count = 0; //Counts number of iterations
		int desiredOutput, output; 	//Calculated output and expected output
		double gError, lError = 0;	//error values
		randomWeights(); //Set weights to initial random values

		do{ 
			gError = 0;
			count++;
			for(int i = 40; i < x1.length; i++){ //Test each data point
				output = calculateOutput(x1[i], x2[i], x3[i], x4[i]); //determine calculated output
				desiredOutput = calculateDesiredOutputAdaline(name[i]); //determine real output
				lError = desiredOutput - output;
				//update weight values
				w[0] += learningRate * lError * x1[i];
				w[1] += learningRate * lError * x2[i];
				w[2] += learningRate * lError * x3[i];
				w[3] += learningRate * lError * x4[i];
				w[4] += learningRate * lError;
				gError += (lError * lError)/2; //update error value
							
			}
			
		}while(gError > 0); //iterate until no errors
       System.out.println("\nNumber of iterations for error correction: " +count);     
	}
	
	/**
	 * This function is used to predict if a set of inputs belongs to Iris-versicolor or
	 * to Iris-virginica based on the weights set in the trainErrorCorrection algorithm.
	 * @throws IOException
	 */
	public void testErrorCorrection() throws IOException{
		double testx1[] = new double[20];
		double testx2[] = new double[20];
		double testx3[] = new double[20];
		double testx4[] = new double[20];
		String[] testName = new String[20];
	    String[] line;
	    //Set arrays for test data points
	    for(int i = 0; i < testx1.length; i++){
	    	line = testValues[i+10].split(",");
	    	testx1[i] = Double.parseDouble(line[0]);
	    	testx2[i] = Double.parseDouble(line[1]);
	    	testx3[i] = Double.parseDouble(line[2]);
	    	testx4[i] = Double.parseDouble(line[3]);
	    	testName[i] = line[4];
	    }
	    errorTest = new String[20];
	    for(int i = 0; i < testx1.length; i++){ //Loop through all data points and test
	    	int output = calculateOutput(testx1[i], testx2[i], testx3[i], testx4[i]);
	    	if(output == 1) //Output of 1 refers to Versicolor
	    		errorTest[i] = "Iris-Versicolor";
	    	else
	    		errorTest[i] = "Iris-Virginica";
	    }
	}
	
	
	/**
	 * This function uses a simple linear seperator to determine if a set of inputs belongs
	 * to Iris-setosa or not. 
	 * @throws NumberFormatException
	 * @throws IOException
	 */
	public void Setosa() throws NumberFormatException, IOException{
        for(int i = 0; i < x1.length; i++) {
      		double y = 1.5 - (1.5 / 4.0) * x3[i]; //compute value on line 
      		if(x4[i] < y) //check if point is above or below line
      			System.out.println("yes"); //point is below, it is Setosa
      		else
      			System.out.println("no"); //point is above, not Setosa
       }
	}
	
	
	/**
	 * 
	 * @return Returns 1 or 0 depending if the sum is greater than theta or not
	 */
	public int calculateOutput(double x1, double x2, double x3, double x4)
	{
		double sum = x1 * w[0] + x2 * w[1] + x3 * w[2] + x4 * w[3] + w[4];
		return (sum >= theta) ? 1 : 0;
	}
	
	
	/**
	 * 
	 * @param name Name of the flower species
	 * @return Returns 1 if sestosa, 0 otherwise
	 */
	public int calculateDesiredOutput(String name){  
		if(name.charAt(5) == 's') //For some reason my program was having trouble equating entire word
			return 1;
		else
			return 0;
	}
	
	
	/**
	 * 
	 * @param name Name of flower species
	 * @return 1 if versicolor, 0 if virginica
	 */
	public int calculateDesiredOutputAdaline(String name){
		if(name.charAt(6) == 'e') //For some reason my program was having trouble equating entire word
			return 1;
		else
			return 0;
	}
	
	
	/**
	 * This function generates random weights for starting values
	 */
	public void randomWeights(){
		for(int i = 0; i < 5; i++){
			w[i] = Math.random(); 
		}
	}
	
	
	/**
	 * This function writes the real and predicted outputs of the two algorithms
	 * to a text file named "Outputs.txt"
	 * @throws IOException
	 */
	public void outputFile() throws IOException{
		BufferedWriter out = new BufferedWriter(new FileWriter("Outputs.txt"));
		out.write("Testing for perceptron: \n");
		for(int i = 0; i < testValues.length; i++){
			String[] name = testValues[i].split(",");
			out.write("\nReal Value: ");
			out.write(name[4]);
			out.write("\tPredicted Value: ");
			out.write(percepTest[i]);
		}
		out.write("\n\n\nTesting for error correction: \n");
		for(int i = 10; i < testValues.length; i++){
			String[] name = testValues[i].split(",");
			out.write("\nReal Value: ");
			out.write(name[4]);
			out.write("\tPredicted Value: ");
			out.write(errorTest[i - 10]);
		}
		out.close();
	}
}
