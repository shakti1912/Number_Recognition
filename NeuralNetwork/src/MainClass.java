
public class MainClass 
{
	public static void main(String[] args)
	{
		// two matrices for testing matrix manipulations class functions
		double[][] one =  { {1,2,3},
				{4,5,6}
		};
		double[][] two =  { {1},
				{4},
				{7}
				
		};
		
		Matrix_Manipulations mm = new Matrix_Manipulations();
		
		//double[][] res = mm.dotProduct(one, two);	//dot product working fine
		//mm.printMatrix(res);
		
		//mm.printMatrix(two);
		//int[][] t = mm.transpose(two);			//transpose working fine
		//mm.printMatrix(t);
		
		NeuralNetwork nn = new NeuralNetwork(784,100,10,0.5);
		
		//testing different functionalities
		//double[] a = {1.16, 0.42, 0.62};
		//double[] res = nn.sigmoidFunction(a);
		/*
		for(int i = 0; i < res.length; i++)
		{
			System.out.print(res[i] + " ");
		}
		 */
		//double[] x = nn.readCSV();
		//mm.printArray(x);
		//System.out.println(x.length);
		
		
		//mm.printMatrix(nn.W_input_hidden);
		
		//mm.printMatrix(nn.query());
		
		//nn.train();
		
		
		//testing query function
		double[][] x = nn.query();
		mm.printMatrix(x);
		
		
		//training for first 100 values from the csv file.
		int ToTrainON = 100;	// we can change this to train on all 60,000 sets
		for(int i = 0; i < ToTrainON; i++)
		{
			nn.train(); 
		}
		
		// Now here is the testing from the mnist_test.csv file
		// we can change this to test on all 10,000 sets
		int ToTestOn = 10;	// just change this variable 1 or 2 to test on only first or second lines in the csv file.
		for(int i = 0; i < ToTestOn; i++)
		{
			nn.test(); 
		}
		



	}
}
