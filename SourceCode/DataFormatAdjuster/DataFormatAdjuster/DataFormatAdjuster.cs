/* DataFormatAdjuster.cs by Garrisen Cizmich (Cizlin Cizliano).
 * Created 4/25/2020.
 * 
 * This program takes in .csv data and outputs it in a format ready to insert in the DecisionTree classifier. */

using System;
using System.Collections.Generic;
using System.IO;

namespace DataFormatAdjuster
{
	class DataFormatAdjuster
	{
		static int Main(string[] args)
		{
			Console.WriteLine();
			if (args.Length == 0 || (args[0] != "-t" && args[0] != "-c")) // If no files were specified.
			{
				Console.WriteLine("Usage: DataFormatAdjuster <option> <inputFile1> [<inputFile2>] ... [<inputFileN>] <outputDirectory>");
				Console.WriteLine();
				Console.WriteLine("Options:");
				Console.WriteLine("\t-t: Generates training (.in) and test (.txt) data from .csv. Mark class with header \"class\" in .csv.");
				Console.WriteLine("\t-c: Format input data to classify as .txt (no attributes considered to be class).");
				Console.WriteLine();
				Console.WriteLine("For more information on a particular option, use \"DataFormatAdjuster <option>\".");
				Console.WriteLine();
				return 1;
			}
			else if (args.Length == 1) // For more information on a specific option.
			{
				if (args[0] == "-t")
				{
					Console.WriteLine("Usage: DataFormatAdjuster -t <inputFile1> [<inputFile2>] ... [<inputFileN>] <outputDirectory>");
					Console.WriteLine();
					Console.WriteLine("Generates training (.in) and test (.txt) data from .csv. Mark class with header \"class\" in .csv.");
					Console.WriteLine("Output filename format for training data is <inputFilename>_Trainer.in.");
					Console.WriteLine("Output filename format for test data is <inputFilename>_Tester.txt");
					Console.WriteLine();
					Console.WriteLine("Training data is mutually exclusive with test data. Training data selected with proportional bootstrapping.");
					Console.WriteLine("Boostrapping used to select set of tuples from each class to maintain relative proportions in training data.");
					Console.WriteLine("Test data contains remaining data not selected by bootstrap.");
				}
				else if (args[0] == "-c")
				{
					Console.WriteLine("Usage: DataFormatAdjuster -c <inputFile1> [<inputFile2>] ... [<inputFileN>] <outputDirectory>");
					Console.WriteLine();
					Console.WriteLine("Generates classifier-ready data from .csv.");
					Console.WriteLine("Output filename format is <inputFilename>_ToClassify.txt.");
					Console.WriteLine();
					Console.WriteLine("All data from .csv is transferred to the output .txt file in classifier-ready format.");
				}
				Console.WriteLine();
				return 1;
			}
			else
			{
				try
				{
					for (int z = 1; z < args.Length - 1; ++z)
					{
						Console.WriteLine("Formatting data from file " + Path.GetFileName(args[z]));
						string[] fileLines = File.ReadAllLines(args[z]); // The lines in the .csv file.
						string[] attributeNames = fileLines[0].Split(','); // The attribute names.

						// Trim the excess whitespace and make the attribute names legal.
						for (int i = 0; i < attributeNames.Length; ++i)
						{
							attributeNames[i] = attributeNames[i].Trim().Replace(' ', '_');
						}

						Dictionary<string, string>[] tuples = new Dictionary<string, string>[fileLines.Length - 1]; // An array of tuples.
						Dictionary<string, List<string>> possibleAttributeValues = new Dictionary<string, List<string>>(); // The possible values for each attribute.
						Dictionary<string, List<int>> classIndices = new Dictionary<string, List<int>>(); // The indices corresponding to tuples in a given class.


						for (int i = 0; i < tuples.Length; ++i)
						{
							tuples[i] = new Dictionary<string, string>();
						}

						for (int i = 1; i < fileLines.Length; ++i)
						{
							string[] tupleValues = fileLines[i].Split(','); // The values of the tuple.
							for (int j = 0; j < tupleValues.Length; ++j)
							{
								// Remove excess whitespace and make values legal.
								tupleValues[j] = tupleValues[j].Trim().Replace(' ', '_');
								// For every attribute name and value, we need to avoid internal whitespace and trim external whitespace.
								tuples[i - 1].Add(attributeNames[j], tupleValues[j]);

								if (i == 1)
								{
									possibleAttributeValues.Add(attributeNames[j], new List<string>());
									possibleAttributeValues[attributeNames[j]].Add(tupleValues[j]);
								}

								// Add the attribute value to the possibilities list if necessary.
								if (i > 1 && !possibleAttributeValues[attributeNames[j]].Contains(tupleValues[j]))
								{
									possibleAttributeValues[attributeNames[j]].Add(tupleValues[j]);
								}

								// We need to add the index to the classIndices dictionary for the particular value of the class.
								if (attributeNames[j].ToLower() == "class")
								{
									if (classIndices.ContainsKey(tupleValues[j]))
									{
										classIndices[tupleValues[j]].Add(i - 1);
									}
									else
									{
										classIndices.Add(tupleValues[j], new List<int>() { i - 1 });
									}
								}
							}
						}

						// Now, we have the tuples in the Dictionary array. The desired file format is as follows:
						// Assuming -t:
						// Number of attributes we are keeping
						// attr1Name attr1Val1 attr1Val2 ...
						// ...
						// Ans class1 class2 ...
						// tuple1Attr1Val tuple1Attr2Val ... tuple1Class
						// ...

						// Assuming -c:
						// attr1Name attr2Name ...
						// tuple1Attr1Val tuple1Attr2Val ...
						// ...

						// First, we consider how many attributes we'll actually need. We have the following attributes:
						// 0. ASIN (ignore)
						// 1. Product_ID (ignore)
						// 2. Product_Group
						// 3. Sales_Rank (continuous)
						// 4. Estimated_Number_Of_Sales (continuous)
						// 5. Purchase_Price (continuous)
						// 6. Buy_Box_Landed (continuous)
						// 7. Low_New_Fba_Price (continuous)
						// 8. Low_New_Mfn_Price (continuous)
						// 9. Sell_Price (continuous)
						// 10. Fulfillment_Subtotal (continuous)
						// 11. Cost_Sub_Total (continuous)
						// 12. Inbound_Shipping_Estimate (continuous)
						// 13. Package_Weight (continuous)
						// 14. Package_Height (ignore)
						// 15. Package_Length (ignore)
						// 16. Package_Width (ignore)
						// 17. Package_Quantity (continuous)
						// 18. Total_Offers (continuous)
						// 19. New_FBA_Num_Offers (continuous)
						// 20. New_MFN_Num_Offers (continuous)
						// 21. Min (ignore)
						// 22. Mult (ignore)
						// 23. UM (ignore)
						// 24. CA (ignore)
						// 25. BWOT (ignore)
						// 27. Class (the tuple's class, only for training data).

						if (args[0] == "-t")
						{
							// Change the following two variables as needed to generate the appropriate test data. All continuous and ignored attributes must be marked here.
							List<int> continuousAttributeIndices = new List<int>()
						{ 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20 }; // The indices (see above) of the continuous attributes.

							List<int> ignoredAttributeIndices = new List<int>()
						{ 0, 1, 14, 15, 16, 21, 22, 23, 24, 25 }; // The indices (see above) of the ignored attributes.

							// We will create two files: one for training data and one for testing data.
							// We will use bootstrapping to randomly select tuples with replacement. 
							// The number selected for each class type will be equal to the number of tuples.
							List<Dictionary<string, string>> trainingTuples = new List<Dictionary<string, string>>(); // The tuples in the training file.
							List<Dictionary<string, string>> testTuples = new List<Dictionary<string, string>>(); // The tuples in the test file.
							Random random = new Random(); // Used to randomly select a tuple.

							// Select the training tuples for each class so that a roughly proportional weighting appears in the final training set.
							foreach (string classValue in classIndices.Keys)
							{
								// Populate the training set (roughly 63% of the data for each class will fall in this set).
								for (int i = 0; i < classIndices[classValue].Count; ++i)
								{
									int randTemp = random.Next(0, classIndices[classValue].Count); // The random integer for this iteration.
									if (!trainingTuples.Contains(tuples[classIndices[classValue][randTemp]]))
									{
										trainingTuples.Add(tuples[classIndices[classValue][randTemp]]);
									}
								}
							}

							// Populate the test set (the remaining tuples fall in this set).
							for (int i = 0; i < tuples.Length; ++i)
							{
								if (!trainingTuples.Contains(tuples[i]))
								{
									testTuples.Add(tuples[i]);
								}
							}


							// Now, we write the training data file.
							string outputFileName = string.Empty; // The name of the output file.

							// Make sure we handle backslashes that may or may not be at the end of the directory.
							if (args[^1][^1] != '\\')
							{
								outputFileName = args[^1] + "\\" + Path.GetFileNameWithoutExtension(args[z]) + "_Trainer.in";
							}
							else
							{
								outputFileName = args[^1] + Path.GetFileNameWithoutExtension(args[z]) + "_Trainer.in";
							}

							Directory.CreateDirectory(Path.GetDirectoryName(outputFileName));

							StreamWriter outputFile = new StreamWriter(File.Create(outputFileName)); // Create the StreamWriter object to write to the file.

							int classIndex = -1; // The index of the class. It will be assigned within the attribute loop.

							// Write the number of attributes to the file (exclude the class).
							outputFile.WriteLine(attributeNames.Length - 1);

							for (int i = 0; i < attributeNames.Length; ++i)
							{
								// We print all of the non-class attributes first. This solution allows us to make any attribute the class by simply renaming it "class".
								if (attributeNames[i].ToLower() != "class")
								{
									outputFile.Write(attributeNames[i]);
									if (continuousAttributeIndices.Contains(i))
									{
										outputFile.WriteLine(" continuous");
									}
									else if (ignoredAttributeIndices.Contains(i))
									{
										outputFile.WriteLine(" ignore");
									}
									else
									{
										// Write each of the possible attribute values.
										foreach (string attributeValue in possibleAttributeValues[attributeNames[i]])
										{
											outputFile.Write(" " + attributeValue);
										}
										outputFile.WriteLine();
									}
								}
								else
								{
									classIndex = i;
								}
							}

							// Write the class and its values.
							outputFile.Write("Ans");
							foreach (string classValue in possibleAttributeValues[attributeNames[classIndex]])
							{
								outputFile.Write(" " + classValue);
							}
							outputFile.WriteLine();

							// Write the tuples.
							for (int i = 0; i < trainingTuples.Count; ++i)
							{
								// Write each of the non-class attributes.
								foreach (string attributeName in attributeNames)
								{
									if (attributeName.ToLower() != "class")
									{
										outputFile.Write(trainingTuples[i][attributeName] + " ");
									}
								}

								// Write the class.
								outputFile.WriteLine(trainingTuples[i][attributeNames[classIndex]]);
							}

							// Flush the buffer and close the file.
							outputFile.Flush();
							outputFile.Close();

							Console.WriteLine("Formatted training data written to " + outputFileName);

							// Now we write the test data file (which is in the same format as the -c option output).
							// Make sure we handle backslashes that may or may not be at the end of the directory.
							if (args[^1][^1] != '\\')
							{
								outputFileName = args[^1] + "\\" + Path.GetFileNameWithoutExtension(args[z]) + "_Tester.txt";
							}
							else
							{
								outputFileName = args[^1] + Path.GetFileNameWithoutExtension(args[z]) + "_Tester.txt";
							}

							Directory.CreateDirectory(Path.GetDirectoryName(outputFileName));

							outputFile = new StreamWriter(File.Create(outputFileName)); // Create the StreamWriter object to write to the file.

							// Write the attribute names at the top.
							foreach (string attributeName in attributeNames)
							{
								outputFile.Write(attributeName + " ");
							}
							outputFile.WriteLine();

							// Write each tuple on its own line.
							foreach (Dictionary<string, string> tuple in testTuples)
							{
								foreach (string attributeName in attributeNames)
								{
									outputFile.Write(tuple[attributeName] + " ");
								}
								outputFile.WriteLine();
							}

							// Flush the buffer and close the file.
							outputFile.Flush();
							outputFile.Close();

							Console.WriteLine("Formatted test data written to " + outputFileName);
						}
						else // The classifying data is much easier to format.
						{
							string outputFileName = string.Empty; // The name of the output file.

							// Make sure we handle backslashes that may or may not be at the end of the directory.
							if (args[^1][^1] != '\\')
							{
								outputFileName = args[^1] + "\\" + Path.GetFileNameWithoutExtension(args[z]) + "_ToClassify.txt";
							}
							else
							{
								outputFileName = args[^1] + Path.GetFileNameWithoutExtension(args[z]) + "_ToClassify.txt";
							}

							Directory.CreateDirectory(Path.GetDirectoryName(outputFileName));

							StreamWriter outputFile = new StreamWriter(File.Create(outputFileName)); // Create the StreamWriter object to write to the file.

							// Write the attribute names at the top.
							foreach (string attributeName in attributeNames)
							{
								outputFile.Write(attributeName + " ");
							}
							outputFile.WriteLine();

							// Write each tuple on its own line.
							foreach (Dictionary<string, string> tuple in tuples)
							{
								foreach (string attributeName in attributeNames)
								{
									outputFile.Write(tuple[attributeName] + " ");
								}
								outputFile.WriteLine();
							}

							// Flush the buffer and close the file.
							outputFile.Flush();
							outputFile.Close();

							Console.WriteLine("Formatted data written to " + outputFileName);
						}
					}
					return 0;
				}
				catch (Exception ex)
				{
					Console.WriteLine("Data formatting failed! " + ex.Message);
					return 1;
				}
			}
		}
	}
}
