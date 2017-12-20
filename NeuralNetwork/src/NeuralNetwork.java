import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.Random;
import java.util.Scanner;

/*
 * Neural Network class with all the playing with data.
 */
public class NeuralNetwork 
{
	private Matrix_Manipulations mxFunctions;
	private int inodes;
	private int hnodes;
	private int onodes;
	private double lr;
	private double[][] X_hidden;	//before sigmoid
	private double[][] O_hidden;	//after sigmoid
	private double[][] X_output;
	private double[][] O_output;
	private double result = -1; // this is the result of our neural network. Initialing it to -1 because answer is between 0 to 9
	private int readTillNow = 0;
	private int testedTillNow = 0;	

	//these are link weights. we will refine these weights by training NN. 
	double[][] W_input_hidden;					//weights between input and hidden => 𝑊_𝑖𝑛𝑝𝑢𝑡_ℎ𝑖𝑑𝑑𝑒𝑛_, of size (hidden_nodes​ by input_nodes).
	double[][] W_hidden_output;				// weights between hidden and output => 𝑊_​ℎ𝑖𝑑𝑑𝑒𝑛_𝑜𝑢𝑡𝑝𝑢𝑡_​, of size (output_nodes​ by hidden_nodes).

	private double[][] inputs;

	public NeuralNetwork(int inputnodes, int hiddennodes, int outputnodes, double learningrate)
	{

		this.inodes = inputnodes;	//784
		this.hnodes = hiddennodes; //100
		this.onodes = outputnodes;	//10
		this.lr = learningrate;	//0.3
		setLinkWeights();
		mxFunctions = new Matrix_Manipulations();

	}

	/*
	 * Initialing weight matrices 
	 * W_input_hidden and W_hidden_output
	 */
	public void setLinkWeights()
	{
		// W_input_hidden is 100 * 784 so we need to make the range accordingly.
		// 100 * 784 => range : +1/sqroot(784) to -1/sqroot(784)

		W_input_hidden = new double[this.hnodes][this.inodes];
		Random generator = new Random();
		double m = 1/(Math.sqrt(784));
		int r = (int)(m * 1000);
		double max = r/1000.0;
		double min = -max;
		//System.out.println(max);
		//System.out.println(generator.nextDouble());
		//System.out.println(min + (generator.nextDouble() * 2* max));
		for(int i = 0 ; i < W_input_hidden.length; i++)
		{
			for( int j = 0 ; j < W_input_hidden[0].length; j++)
			{

				W_input_hidden[i][j] = min + (generator.nextDouble() * 2* max); // min = -1/sqroot(784)
																				// max = +1/sqroot(784)
			}
		}

		// W_hidden_output is 10 * 100 so we need to make the range accordingly.
		// 10 * 100 => range : -1/sqroot(100) to +1/sqroot(100)

		W_hidden_output = new double[this.onodes][this.hnodes];

		double m2 = 1/(Math.sqrt(100));
		int r2 = (int)(m2 * 1000);
		//System.out.println(r2);
		double max2 = r2/1000.0;
		double min2 = -max2;

		for(int i = 0 ; i < W_hidden_output.length; i++)
		{
			for( int j = 0 ; j < W_hidden_output[0].length; j++)
			{					
				W_hidden_output[i][j] = (min2 + (generator.nextDouble() * 2* max2));		// min2 = -1/sqroot(100)
																							// max2 = +1/sqroot(100)
			}
		}

	}

	/*
	 * Training data using training file
	 */
	public void train()
	{
		
		//first query 
		double[][] output = query();
		double[][] E_output = this.error(output);  //e_k


		//second, now backpropagate. Calculate derivative part
		// dE/dw = -(t_k - o_k)*o_k*(1-o_k)*oT_j
		double[][] part_one = new double[E_output.length][1];
		for(int i = 0 ; i < E_output.length; i++)
		{
			// (t_k - o_k) = e_k (calculated above),  o_k = X_output 
			part_one[i][0] =  E_output[i][0] * O_output[i][0] *(1 - O_output[i][0]); //o_k = X_output which is matrix before 
			// final activation function is applied but using X_output because here it is used in place of sigmoid(X_output) which is O_output.
		}
		//transpose output matrix from previous layer which is O_hidden (output from hidden layer)
		double[][] O_output_transpose = mxFunctions.transpose(O_hidden);
		//calculating derivative
		double[][] change_in_weights = mxFunctions.dotProduct(part_one, O_output_transpose);

		//calculating new weights or modifying weight matrices to update neural network
		//here we are updating W_hidden_output by subtracting old weight - (learningrate * derivative value just calculated.)
		for(int i = 0 ; i < W_hidden_output.length; i++)
		{
			for(int j = 0 ; j < W_hidden_output[0].length; j++)
			{
				W_hidden_output[i][j] = W_hidden_output[i][j] - (this.lr * change_in_weights[i][j]);
			}
		}

		//NOW we need to do same procedure for 1st part of neural network meaning weights between input nodes and hidden nodes
		
		//first calculating error_hidden which is dot product of transpose of weights(between input nodes and hidden nodes) and output error matrix.
		
		double[][] E_hidden = mxFunctions.dotProduct(mxFunctions.transpose(W_hidden_output) ,this.error(output));  //e_j
		

		//second, now backpropagate. Calculate derivative part
		// dE/dw = -(t_j - o_j)*o_j*(1-o_j)*oT_i  //o_i is basically the input signals
		double[][] part_one1 = new double[E_hidden.length][1];
		for(int i = 0 ; i < E_hidden.length; i++)
		{
			////o_j = X_hidden which is matrix before 
			// final activation function is applied but using X_output because here it is used in place of sigmoid(X_hidden) which is X_output.
			part_one1[i][0] =  E_hidden[i][0] * O_hidden[i][0] *(1 - O_hidden[i][0]); //o_i = inputs matrix		
		}
		//transpose inputs matrix which is o_i
		double[][] inputs_transpose = mxFunctions.transpose(inputs);
		//calculating derivative
		double[][] change_in_weights2 = mxFunctions.dotProduct(part_one1, inputs_transpose);

		//calculating new weights or modifying weight matrices to update neural network
		//here we are updating W_input_hidden by subtracting old weight - (learningrate * derivative value just calculated.)
		for(int i = 0 ; i < W_input_hidden.length; i++)
		{
			for(int j = 0 ; j < W_input_hidden[0].length; j++)
			{
				W_input_hidden[i][j] = W_input_hidden[i][j] - (this.lr * change_in_weights2[i][j]);
			}
		}
	}

	/*
	 * Querying neural network
	 */
	public double[][] query()
	{
		//first read data from csv file. This will set variable inputs rescaled after reading from file.
		readCSV();
		readTillNow++; // updating the line till we have read in csv file so that next time following line is read from csv file.
		//we have 3 layers
		//first layer work 
		// _𝑿_​ℎ𝑖𝑑𝑑𝑒𝑛_​ ​= 𝑾_​_𝑖𝑛𝑝𝑢𝑡_ℎ𝑖𝑑𝑑𝑒𝑛_​ · 𝑰  => this goes into sigmoid function to get output from first layer
		X_hidden = mxFunctions.dotProduct(W_input_hidden, inputs);
		// 𝑶_​_ℎ𝑖𝑑𝑑𝑒𝑛_​ ​= 𝑠𝑖𝑔𝑚𝑜𝑖𝑑( _𝑿_​ℎ𝑖𝑑𝑑𝑒𝑛_ ​)
		O_hidden = this.sigmoidFunction(X_hidden);

		//second layer work 
		// _𝑿_​output_​ ​= 𝑾_​hidden_output​ · 𝑰  => this goes into sigmoid function to get output from second layer which is the actual output.
		X_output = mxFunctions.dotProduct(W_hidden_output, O_hidden);
		//second 𝑶_​_output_​ ​= 𝑠𝑖𝑔𝑚𝑜𝑖𝑑( _𝑿_​output_ ​)
		O_output = this.sigmoidFunction(X_output);

		return O_output;		//this is the output matrix that is calculated for each query

	}

	/*
	 * Input data
	 */
	public double[][] readCSV()
	{
		File file = new File("mnist_train_100.csv");
		Scanner in = null;
		double[][] inputData = null;
		try
		{
			in = new Scanner(file);
			for(int i = 0; i < readTillNow; i++)
			{
				in.nextLine();
			}
			
			String line = in.nextLine();
			//System.out.print(line);
			double[] data = parseIntCus(line.split(","));
			//convert them to 0.01 to 1.0
			inputData =  rescale(data);
			//}
		}

		catch (FileNotFoundException e) 
		{
			e.printStackTrace();
		}
		return inputData;


	}
	/*
	 * Rescale input data and convert them into matrix. 
	 * This is a 2D array but with rows = nodes and column = 1
	 */
	public double[][] rescale(double[] a)
	{
		double[][] scaledData = new double[784][1];
		result = a[0];
		for(int i = 1; i < a.length; i++)	//skip first which is result
		{
			//scaling values to avoid saturation of neural network and also avoiding 0 
			//because it will lead to no result making all the calculations 0.
			double scaled = (a[i]/255)*0.99 + 0.01;
			scaledData[i-1][0] = scaled;

		}
		inputs = scaledData;
		return scaledData;
		/*
		for(int i=0; i< 28; i++)
		{
			for(int j=0; j < 28; j++)
			{
				double scaled = (a[(i*28 + j)]/255)*0.99 + 0.01;
				scaledData[i][j] = scaled; 
			}
		}
		return scaledData;
		 */
	}

	private double[] parseIntCus(String[] split) {
		// TODO Auto-generated method stub
		double[] res = new double[split.length];
		for(int i = 0; i < split.length; i++)
		{
			res[i] = Integer.parseInt(split[i]);
		}
		return res;
	}
	/*
	 * Sigmoid function
	 */
	public double[][] sigmoidFunction(double[][] X)
	{
		double[][] sigResult = new double[X.length][1];

		for(int i = 0; i < X.length; i++)
		{
			sigResult[i][0] = (1 / ( 1 + Math.pow(Math.E, -(X[i][0]))));

		}
		return sigResult;
	}

	/*
	 * This function calculates the error matrix
	 * it is : error = trainingData - output
	 * 
	 */
	public double[][] error(double[][] output)
	{
		double[][] error = new double[this.onodes][1];
		
		//The training data should be this for result = 5: 
		// [0.01, 0.01, 0.01, 0.01, 0.01, 0.99, 0.01, 0.01, 0.01, 0.01] 
		//here I making new trainingData matrix that depending on result makes the kind of matrix shown above.
		double[][] trainingData = new double[this.onodes][1];
		for(int i = 0; i < this.onodes; i++)
		{
			if(i == result)
			{
				trainingData[i][0] = 0.99;
			}
			else
			{
				trainingData[i][0] = 0.01;		
			}
		}
		//Here I am doing e_k = t_k - o_k
		for(int i = 0; i < this.onodes; i++)
		{
			error[i][0] = trainingData[i][0] - output[i][0];
		}
		return error;

	}

	/*
	 * this is test function that reads from test csvfile and present the output.
	 * Since we able trained our Neural Network, we just need to query for each
	 * set in this file and see if it is (probably) correct or not.
	 */
	public void test() 
	{
		File file = new File("mnist_test_100.csv");
		testedTillNow++;
		Scanner in = null;
		double[][] inputData = null;
		try
		{
			in = new Scanner(file);
			for(int i = 0; i < testedTillNow; i++)
			{
				in.nextLine();
			}
			String line = in.nextLine();
			//System.out.print(line);
			double[] data = parseIntCus(line.split(","));
			//convert them to 0.01 to 1.0
			inputData =  rescale(data);
			//inputData is same as inputs matrix now.
			//now just query the network to test the network
			this.query();
			
		}

		catch (FileNotFoundException e) 
		{
			e.printStackTrace();
		}
		
	}

}
