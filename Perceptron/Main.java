import java.io.IOException;

/**
 * This code is simply used to generate a PerceptronFL object, and to run its functions.
 * @author patrickgibson
 *
 */
public class Main {

	public static void main(String[] args) {
		try {
			PerceptronFL percep = new PerceptronFL();
			percep.Setosa();							//Part 1
			percep.trainPerceptron();					//Train perceptron algorithm
			percep.testPerceptron();					//Test with set weights values
			percep.trainErrorCorrection();				//Train error correction algorithm
			percep.testErrorCorrection();				//Test with set weight values
			percep.outputFile();						//Output info to text file
		} catch (IOException e) {
			e.printStackTrace();
		}	
	}	
}
