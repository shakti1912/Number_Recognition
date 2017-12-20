/*
 * Class with methods for matrix manipulations
 */
public class Matrix_Manipulations 
{

	/**
	 * multiply two regular matrices
	 * @param a given matrix
	 * @param b given matrix
	 * @return returns regular product 
	 */
	public  double[][] dotProduct(double[][] a, double[][] b )
	{
		int rowOfA = a.length;
		int columnOfA = a[0].length;
		int rowOfB = b.length;
		int columnOfB = b[0].length;

		if (columnOfA != rowOfB) 
		{
			System.out.println("Illegal Matrix Dimensions");
			return null;
		}	

		double[][] C = new double[rowOfA][columnOfB];
		for (int i=0; i<rowOfA; i++)
		{
			for (int j=0; j<columnOfB; j++)
			{
				for (int k=0; k<columnOfA; k++)
				{
					C[i][j] += a[i][k] * b[k][j];
				}
			}
		}	
		return C;
	}

	

	/*
	 * Transpose of a matrix
	 */
	public double[][] transpose(double[][] a)
	{
		double[][] trans = new double[a[0].length][a.length];

		
		for (int i = 0; i < a.length; i++) 
		{	
			for (int j = 0; j < a[0].length; j++) 
			{

				trans[j][i] = a[i][j];
		
			}
		}
		return trans;
	}

	/*
	 * 
	 * Print a given matrix
	 */
	public void printMatrix(double[][] x)
	{
		for(int i = 0; i < x.length; i++)
		{
			for (int j = 0; j < x[0].length; j++)
			{
				System.out.print(x[i][j] + " ");

			}
			System.out.println();

		}
	}
	/*
	 * 
	 * Print a given array
	 */
	public void printArray(double[] x)
	{
		for(int i = 0; i < x.length; i++)
		{
			
				System.out.print(x[i] + " ");

		}
	}
}
