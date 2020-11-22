/* DecisionTree.cs by Garrisen Cizmich (known by some as Grasmussen)
 * Collaborated with Austin White and Aleesha Chavez to generate the classification model.
 * Collaborated with Aleesha Chavez and Enoch Levandovsky to write the model to file, retrieve it from file, and classify tuples.
 * Modified 4/10/2020 by Garrisen Cizmich in collaboration with Aleesha Chavez and Enoch Levandovsky.
 * 
 * This program originally took in training data, created a decision tree in memory, and used DFS to output the tree to console.
 * Now, it gives the user the ability to train a classifier with training data, load a classifier from memory, and classify tuples.
 * The attribute selection method used is information gain. */
using System;
using System.Collections.Generic;
using System.IO;

namespace DecisionTree
{
	/// <summary>
	/// A node for use with a decision tree. Contains two data members: NextAttribute and Rule.
	/// NextAttribute specifies the the next attribute to check for a given rule.
	/// Rule specifies the rule by which we reach the node.
	/// The node also stores a list of TreeNodes which are its children.
	/// </summary>
	public class TreeNode
	{
		/// <summary>
		/// Initializes the node with a rule and attribute.
		/// </summary>
		/// <param name="rule">The rule by which we reach the node.</param>
		/// <param name="nextAttribute">The next attribute to check (or the class).</param>
		public TreeNode(string rule, string nextAttribute)
		{
			Rule = rule;
			NextAttribute = nextAttribute;
		}

		/// <summary>
		/// Adds a child to the node's list with the given rule and attribute values.
		/// </summary>
		/// <param name="rule">The rule by which we reach this node.</param>
		/// <param name="nextAttribute">The next attribute to check (or the class).</param>
		public void AddChild (string rule, string nextAttribute)
		{
			Children.Add(new TreeNode(rule, nextAttribute));
		}

		/// <summary>
		/// The children of the node.
		/// </summary>
		public List<TreeNode> Children { get; } = new List<TreeNode>();

		/// <summary>
		/// The rule by which we get to the node. 
		/// </summary>
		public string Rule { get; set; }

		/// <summary>
		/// The next attribute for comparison. Can also be the class of the node.
		/// </summary>
		public string NextAttribute { get; set; }
	}

	/// <summary>
	/// A tree object intended for building a decision tree.
	/// </summary>
	public class Tree
	{
		/// <summary>
		/// Creates a tree object with a new root, holding the rule and nextAttribute specified.
		/// </summary>
		/// <param name="rule">This rule should be empty on the root of the tree.</param>
		/// <param name="nextAttribute">The first attribute on which we split.</param>
		public Tree (string rule, string nextAttribute)
		{
			Root = new TreeNode(rule, nextAttribute);
		}

		/// <summary>
		/// The root of the tree. Immutable.
		/// </summary>
		public TreeNode Root { get; }
	}

	/// <summary>
	/// An object containing the name and values of an attribute. Can also be used to store the classes.
	/// </summary>
	public class Attribute
	{
		/// <summary>
		/// Creates an attribute from a name and array of values.
		/// </summary>
		/// <param name="name">The name of the attribute.</param>
		/// <param name="values">The array containing the possible values of the attribute.</param>
		public Attribute(string name, string[] values)
		{
			Name = name;
			Values = new string[values.Length];
			values.CopyTo(Values, 0);
		}

		/// <summary>
		/// The name of the attribute.
		/// </summary>
		public string Name { get; set; }

		/// <summary>
		/// An array of values that the attribute can take.
		/// </summary>
		public string[] Values { get; set; }
	}

	/// <summary>
	/// An object containing a tuple's class and attribute values.
	/// </summary>
	public class Tuple
	{
		/// <summary>
		/// Creates an empty tuple. Data must be added manually.
		/// </summary>
		public Tuple ()
		{
			AttributeValues = new Dictionary<string, string>();
		}

		/// <summary>
		/// Creates a tuple from a class name, an array of attribute names, and an index-matched array of attribute values.
		/// </summary>
		/// <param name="tupleClass">The name of the tuple's class.</param>
		/// <param name="attributeNames">An array containing each attribute's name.</param>
		/// <param name="attributeValues">An array containing each attribute's for the given tuple.</param>
		public Tuple (string tupleClass, string[] attributeNames, string[] attributeValues)
		{
			Class = tupleClass;
			AttributeValues = new Dictionary<string, string>();

			// Add each attribute value to the dictionary.
			for (int i = 0; i < attributeNames.Length; ++i)
			{
				AttributeValues.Add(attributeNames[i], attributeValues[i]);
			}
		}

		/// <summary>
		/// The attribute values, addressable via attribute name.
		/// </summary>
		public Dictionary<string, string> AttributeValues { get; }

		/// <summary>
		/// The tuple's class.
		/// </summary>
		public string Class { get; set;  }
	}

	/// <summary>
	/// This class contains the main functions for reading in training data from a file and creating a decision tree from the data.
	/// It also prints the data to the console.
	/// </summary>
	class DecisionTree
	{
		/// <summary>
		/// Info(D) = - sum from i=1 to m of pi * lg(pi) (not pi = 3.14 but pi).
		/// pi = |Ci,D| / |D|
		/// Calculates the entropy of a set of tuples based on the set and the number of classes in the data.
		/// </summary>
		/// <param name="tupleList">The list of tuples in the partition</param>
		/// <param name="tupleClasses">The attribute containing the possible tuple classes</param>
		/// <returns>Info(D)</returns>
		static double ExpectedInfo(List<Tuple> tupleList, Attribute tupleClasses)
		{
			double sum = 0; // The sum of the info that we will return later (sounds familiar).
			for (int i = 0; i < tupleClasses.Values.Length; ++i)
			{
				int classCount = 0;
				for (int j = 0; j < tupleList.Count; ++j)
				{
					if (tupleList[j].Class == tupleClasses.Values[i])
					{
						++classCount;
					}
				}

				// We cannot take the logarithm of 0, so we ignore those cases.
				if (classCount > 0)
				{
					sum += -1 * Math.Log((double)classCount / tupleList.Count, 2) * classCount / tupleList.Count;
				}
			}

			return sum;
		}

		/// <summary>
		/// InfoA(D) = sum from j=1 to v of |Dj|/|D| * Info(Dj)
		/// Calculates the entropy based on a test partition, given the full list of tuples, the lists of each partition, and the tuple classes.
		/// </summary>
		static double ExpectedInfoWithPartition(List<Tuple> fullTupleList, List<Tuple>[] tuplePartitions, Attribute tupleClasses)
		{
			// Implements the function above probably.
			double sum = 0; // The sum of the info that we will return later.
			for (int j = 0; j < tuplePartitions.Length; ++j)
			{
				sum += (double)tuplePartitions[j].Count / fullTupleList.Count * ExpectedInfo(tuplePartitions[j], tupleClasses);
			}

			return sum;
		}

		/// <summary>
		/// Checks to see if the tuples in a sublist are all of the same class.
		/// </summary>
		/// <param name="tupleList">The tuple list to check.</param>
		/// <returns>True if the classes are the same, false otherwise.</returns>
		static bool SubListClassesAreSame(List<Tuple> tupleList)
		{
			string firstTupleClass = tupleList[0].Class;
			foreach (Tuple tuple in tupleList)
			{
				if (tuple.Class != firstTupleClass)
				{
					return false;
				}
			}
			return true;
		}

		/// <summary>
		/// The recursion that builds our tree. Call it after creating a Tree object and pass in the root with empty attributes.
		/// </summary>
		/// <param name="tupleList">The list of tuples we are now considering.</param>
		/// <param name="currentNode">The node to which we will add child nodes.</param>
		/// <param name="attributeList">The list of possible attributes.</param>
		/// <param name="tupleClasses">The classes for the tuples.</param>
		/// <param name="continuousAttributeNames">The list of continuous attributes (necessary for proper printing only).</param>
		static void TreeRecursion(List<Tuple> tupleList, TreeNode currentNode, List<Attribute> attributeList, Attribute tupleClasses, 
			List<string> continuousAttributeNames, List<string> ignoredAttributeNames)
		{
			// If we still have tuples and attributes and the subclasses are not the same, then we need to split based on the best attribute.
			if (tupleList.Count > 0 && attributeList.Count - ignoredAttributeNames.Count > 0 && !SubListClassesAreSame(tupleList))
			{
				string bestAttributeName = string.Empty; // The best attribute for splitting.
				double minExpectedInfo = double.MaxValue; // The minimum expected information for the partition.

				// First we select the best attribute using information gain.
				foreach (Attribute attr in attributeList)
				{
					// If we aren't ignoring the attribute, analyze its usefulness.
					if (!ignoredAttributeNames.Contains(attr.Name))
					{
						Dictionary<string, List<Tuple>> subLists = new Dictionary<string, List<Tuple>>(); // The sublists for our trial partition.
						for (int j = 0; j < attr.Values.Length; ++j)
						{
							subLists.Add(attr.Values[j], new List<Tuple>()); // Create a new sublist object, addressable by the attribute value.
						}

						// Populate the sublists.
						foreach (Tuple tupleBoi in tupleList)
						{
							subLists[tupleBoi.AttributeValues[attr.Name]].Add(tupleBoi); // Add the tuple to the sublist matching its value.
						}

						// We need to convert our dictionary to an array to use our entropy function.
						List<Tuple>[] tupleLists = new List<Tuple>[attr.Values.Length]; // Array version of sublists dictionary object. 
						for (int j = 0; j < tupleLists.Length; ++j)
						{
							tupleLists[j] = subLists[attr.Values[j]];
						}

						double expectedInfo = ExpectedInfoWithPartition(tupleList, tupleLists, tupleClasses); // The expected info needed after making the partition.

						// Update the expected info if necessary.
						if (expectedInfo < minExpectedInfo)
						{
							minExpectedInfo = expectedInfo;
							bestAttributeName = attr.Name;
						}
					}
				}

				// Now that we have the best attribute, we need to create the Node with the current attribute and rule.
				Dictionary<string, List<Tuple>> childLists = new Dictionary<string, List<Tuple>>(); // The sublists for our final partition.

				int bestAttributeIndex = -1; // The index of the best attribute within the attributeList array.

				// The attribute list is index addressed, so we need to get the index associated with the name.
				for (int i = 0; i < attributeList.Count; ++i)
				{
					if (attributeList[i].Name == bestAttributeName)
					{
						bestAttributeIndex = i;
						break;
					}
				}

				currentNode.NextAttribute = bestAttributeName; // The next attribute for determining the rules.

				// Initialize all the sublists.
				for (int j = 0; j < attributeList[bestAttributeIndex].Values.Length; ++j)
				{
					childLists[attributeList[bestAttributeIndex].Values[j]] = new List<Tuple>(); // Create a new sublist object, addressable by the attribute value.
				}

				// Populate the sublists.
				foreach (Tuple tupleBoi in tupleList)
				{
					childLists[tupleBoi.AttributeValues[attributeList[bestAttributeIndex].Name]].Add(tupleBoi); // Add the tuple to the sublist matching its value.
				}

				Attribute toBeRemoved = attributeList[bestAttributeIndex]; // The attribute will be removed before being passed into the recursion.

				// We need to preserve the attribute list in case it is used by another branch of the tree, so we clone it before removing the attribute.
				List<Attribute> prunedAttributeList = new List<Attribute>(attributeList); // List with the currently used attribute removed.

				prunedAttributeList.RemoveAt(bestAttributeIndex); // Remove the attribute we have already used.

				int skippedChildren = 0; // The number of children we did not need to make.

				// Let's do the recursion now!
				for (int j = 0; j < toBeRemoved.Values.Length; ++j)
				{
					// If we don't have any tuples for one of the children, don't even make the distinction.
					if (childLists[toBeRemoved.Values[j]].Count > 0)
					{
						// We perform this check solely to format the strings properly in the tree.
						if (continuousAttributeNames.Contains(toBeRemoved.Name))
						{
							currentNode.AddChild(toBeRemoved.Values[j], string.Empty);
						}
						else
						{
							currentNode.AddChild("=" + toBeRemoved.Values[j], string.Empty);
						}

						// Call the recursion
						TreeRecursion(childLists[toBeRemoved.Values[j]], currentNode.Children[j - skippedChildren], prunedAttributeList, tupleClasses, 
							continuousAttributeNames, ignoredAttributeNames);
					}
					else
					{
						// If no tuples match a particular attribute, then we do not need to make a child for it. Increment the counter to adjust the index.
						++skippedChildren;
					}
				}
			}
			else if (tupleList.Count <= 0) // If there are no tuples left.
			{
				currentNode.NextAttribute = "No such tuples."; // Should never occur now that we don't add any children for empty tuple lists.
			}
			else if (attributeList.Count - ignoredAttributeNames.Count <= 0 && tupleList.Count > 0) // If we ran out of attributes on which to split.
			{
				Dictionary<string, int> classCounts = new Dictionary<string, int>(); // The counts of each class, based on class name.
				foreach (string tupleClass in tupleClasses.Values)
				{
					classCounts.Add(tupleClass, 0); // Add the class types to the dictionary with our baseline count of 0.
				}

				foreach (Tuple tuple in tupleList)
				{
					++classCounts[tuple.Class]; // Increment the counter for each tuple of the desired class.
				}

				int maxCount = int.MinValue; // The maximum count of the mode.
				string maxTupleClass = string.Empty; // The tuple class that is most common.

				// Find the mode, the most frequently occurring tuple class.
				foreach (string tupleClass in tupleClasses.Values)
				{
					if (maxCount < classCounts[tupleClass])
					{
						maxCount = classCounts[tupleClass];
						maxTupleClass = tupleClass;
					}
				}

				currentNode.NextAttribute = maxTupleClass; // We set the class based on majority rule.
			}
			else // We have attributes and tuples, but all of the tuples are of the same class.
			{
				currentNode.NextAttribute = tupleList[0].Class; // Set the class to match the tuples.
			}
		}

		/// <summary>
		/// Writes the contents of the tree to file using a depth-first search.
		/// </summary>
		/// <param name="node">The current node.</param>
		/// <param name="numTabs">The number of tabs to insert before the line.</param>
		/// <param name="outputFile">The stream writer object representing the output file.</param>
		static void WriteToFileDFS(TreeNode node, int numTabs, StreamWriter outputFile)
		{
			// Perform a recursion on each child of the node.
			foreach (TreeNode child in node.Children)
			{
				// Write the proper number of tabs to output the tree.
				for (int i = 0; i < numTabs; ++i)
				{
					outputFile.Write("\t");
				}

				// Write the attribute we would check.
				outputFile.Write(node.NextAttribute);

				// Write the rule that applies at this step.
				outputFile.WriteLine(child.Rule);

				// We now call the function with a number of tabs that is 1 greater than the current number.
				WriteToFileDFS(child, numTabs + 1, outputFile);				
			}

			// If there are no children, then we didn't execute the for loop. We still need to output the class, though.
			if (node.Children.Count == 0)
			{
				// Write the proper number of tabs to output the tree.
				for (int i = 0; i < numTabs; ++i)
				{
					outputFile.Write("\t");
				}

				// Write the class, which is in the next attribute.
				outputFile.WriteLine(node.NextAttribute);
			}
		}


		private static int currentLine = 0; // The index of the current line in the file, needed for the ReconstructTreeDFS function.
		/// <summary>
		/// Rebuilds the decision tree in memory from file. Does the reverse of write operations done in WriteToFileDFS().
		/// </summary>
		/// <param name="node">The node whose children we will create based on the file.</param>
		/// <param name="classifierLines">The lines from the classifier file.</param>
		static void ReconstructTreeDFS(TreeNode node, string[] classifierLines)
		{
			// We will iterate by each line. There are three types of lines: discrete rules, continuous rules, and classifications.
			// Discrete rules only contain an "=" sign.
			// Continous rules either contain a ">" sign or a "<=" sign.
			// Classifications contain neither.
			// We'll handle continuous rules first since they are the easiest to detect.
			while (currentLine < classifierLines.Length)
			{
				if (classifierLines[currentLine].Contains(">"))
				{
					// Continuous rule. Get the attribute name (all text prior to ">").
					string attributeName = classifierLines[currentLine].Trim().Substring(0, classifierLines[currentLine].Trim().IndexOf(">"));

					if (node.NextAttribute == attributeName || node.NextAttribute == string.Empty)
					{
						// Add the attribute to the node.
						node.NextAttribute = attributeName;

						// Now, add a child with the rule and call the recursive function again.
						node.AddChild(classifierLines[currentLine].Trim().Substring(classifierLines[currentLine].Trim().IndexOf(">")), string.Empty);
						++currentLine; // We are finished with this line, move on to the next one.
						ReconstructTreeDFS(node.Children[^1], classifierLines);
					}
					else
					{
						break;
					}
				}
				else if (classifierLines[currentLine].Contains("<"))
				{
					// Continuous rule. Get the attribute name (all text prior to "<").
					string attributeName = classifierLines[currentLine].Trim().Substring(0, classifierLines[currentLine].Trim().IndexOf("<"));

					if (node.NextAttribute == attributeName || node.NextAttribute == string.Empty)
					{
						// Add the attribute to the node.
						node.NextAttribute = attributeName;

						// Now, add a child with the rule and call the recursive function again.
						node.AddChild(classifierLines[currentLine].Trim().Substring(classifierLines[currentLine].Trim().IndexOf("<")), string.Empty);
						++currentLine; // We are finished with this line, move on to the next one.
						ReconstructTreeDFS(node.Children[^1], classifierLines);
					}
					else
					{
						break;
					}
				}
				else if (classifierLines[currentLine].Contains("="))
				{
					// Discrete rule. Get the attribute name (all text prior to "=").
					string attributeName = classifierLines[currentLine].Trim().Substring(0, classifierLines[currentLine].Trim().IndexOf("="));

					if (node.NextAttribute == attributeName || node.NextAttribute == string.Empty)
					{
						// Add the attribute to the node.
						node.NextAttribute = attributeName;

						// Now, add a child with the rule and call the recursive function again.
						node.AddChild(classifierLines[currentLine].Trim().Substring(classifierLines[currentLine].Trim().IndexOf("=")), string.Empty);
						++currentLine; // We are finished with this line, move on to the next one.
						ReconstructTreeDFS(node.Children[^1], classifierLines);
					}
					else
					{
						break;
					}
				}
				else
				{
					// Classification. Write the class to the current node's NextAttribute variable and end the function calls if no more functions need to be read.
					node.NextAttribute = classifierLines[currentLine].Trim();
					++currentLine;
					break;
				}
			}
		}

		/// <summary>
		/// Classifies a tuple based on a decision tree classifier.
		/// </summary>
		/// <param name="node">The current node of the decision tree. For initial calls, use the root.</param>
		/// <param name="tuple">The tuple to be classified.</param>
		static void ClassifyTuples (TreeNode node, Tuple tuple)
		{
			// We check the children based on the attribute in node.NextAttribute.
			// If we see "=", we just compare strings after removing the first and last character of node.Children[i].Rule.
			// If we see "<" or ">", we need to compare floating point values (equivalence goes to "<").

			if (node.Children.Count == 0) // We have arrived at the classification
			{
				tuple.Class = node.NextAttribute; // The class is stored in the NextAttribute 
			}
			else // We have more tree to follow.
			{
				foreach (TreeNode child in node.Children)
				{
                    if (child.Rule[0] == '=')
                    {
                        // We just perform a string comparison for discrete values.
                        string ruleComparison = child.Rule[1..]; // The actual value to compare with the attr. val.
                        if (ruleComparison == tuple.AttributeValues[node.NextAttribute])
                        {
                            ClassifyTuples(child, tuple);
                        }
                    }
                    else if (child.Rule[0] == '<')
                    {
                        // We perform a double comparison for continuous values. Different start index due to "<=" versus "=" or ">".
                        double ruleComparison = double.Parse(child.Rule[2..]); // The actual value to compare with the attr. val.
                        if (Convert.ToDouble(tuple.AttributeValues[node.NextAttribute]) <= ruleComparison)
                        {
							ClassifyTuples(child, tuple);
						}
                    }
                    else if (child.Rule[0] == '>')
                    {
                        // We perform a double comparison for continuous values.
                        double ruleComparison = double.Parse(child.Rule[1..]); // The actual value to compare with the attr. val.
						if (Convert.ToDouble(tuple.AttributeValues[node.NextAttribute]) > ruleComparison)
                        {
                            ClassifyTuples(child, tuple);
                        }
                    }
                }
			}
		}

		/// <summary>
		/// The program takes in file paths as arguments and prints the associated decision trees to the console.
		/// </summary>
		/// <param name="args">The file names, separated by spaces, to be processed.</param>
		/// <returns>0 if working properly, 1 otherwise.</returns>
		static int Main(string[] args)
		{
			Console.WriteLine();
			if (args.Length == 0 || (args[0] != "-t" && args[0] != "-c")) // If no files were specified.
			{
				Console.WriteLine("Usage: DecisionTree <option> <inputFile1> [<inputFile2>] ... [<inputFileN>] <outputDirectory> [<default>]");
				Console.WriteLine();
				Console.WriteLine("Options:");
				Console.WriteLine("\t-t: Train N classifiers based on N sets of input data.");
				Console.WriteLine("\t-c: Classify tuples from N-1 input files. <inputFilePath1> is the file containing the desired classifier.");
				Console.WriteLine("\t    <default> is the default class used when a tuple does not match any path in the tree. Required only for -c.");
				Console.WriteLine();
				Console.WriteLine("For more information on a particular option, use \"DecisionTree <option>\".");
				Console.WriteLine();
				return 1;
			}
			else if (args.Length == 1) // For more information on a specific option.
			{
				if (args[0] == "-t")
				{
					Console.WriteLine("Usage: DecisionTree -t <inputFile1> [<inputFile2>] ... [<inputFileN>] <outputDirectory>");
					Console.WriteLine();
					Console.WriteLine("Generates classifier from .in training data file and writes it to a .txt file.");
					Console.WriteLine("Output filename format is <inputFilename>_Classifier.txt");
					Console.WriteLine();
					Console.WriteLine("ID3 decision tree is constructed based on provided training data.");
					Console.WriteLine("Attributes marked by \"continuous\" are continuous.");
					Console.WriteLine("Attributes marked by \"ignore\" are ignored by the classifier.");
				}
				else if (args[0] == "-c")
				{
					Console.WriteLine("Usage: DecisionTree -c <classifierFile> <inputFile1> [<inputFile2>] ... [<inputFileN>] <outputDirectory> <default>");
					Console.WriteLine();
					Console.WriteLine("Classifies data from formatted .txt input files and writes it to a .csv file.");
					Console.WriteLine("Output filename format is <inputFilename>_Classified.csv.");
					Console.WriteLine();
					Console.WriteLine("The class specified by <default> is used whenever a tuple cannot be classified by the decision tree.");
					Console.WriteLine("This default option must be explicitly specified to run properly.");
					Console.WriteLine("Tuples are reproduced in output file with the class appended on the end under the Ans column.");
				}
				Console.WriteLine();
				return 1;
			}
			else
			{
				if (args[0] == "-t")
				{
					for (int z = 1; z < args.Length - 1; ++z) // Build the decision tree for each file.
					{
						try
						{
							Console.WriteLine("Creating the decision tree based on the input file " + Path.GetFileName(args[z]) + "...");
							Console.WriteLine("Reading in attribute data...");
							// Gather file info and set up attributes.
							string[] inputLines = File.ReadAllLines(args[z]); // The lines from the file.
							List<Attribute> attributeList = new List<Attribute>(); // The list of attributes.
							string[] attributeNames = new string[Convert.ToInt32(inputLines[0])]; // The names of each of the attributes (for the tuple dictionaries).
							List<string> continuousAttributeNames = new List<string>(); // The names of attributes that are continuous.
							Dictionary<string, List<double>> continuousAttributeValues = new Dictionary<string, List<double>>(); // The values for each continuous attribute.
							List<int> continuousIndices = new List<int>(); // The index of each continuous attribute.

							List<string> ignoredAttributeNames = new List<string>(); // The names of attributes we ignore.
							List<int> ignoredIndices = new List<int>(); // The index of each ignored attribute.

							// Populate the attribute list.
							// Each line looks like: name val1 val2 val3 ...
							// First line occurs after initial number of attributes.
							for (int i = 0; i < Convert.ToInt32(inputLines[0]); ++i)
							{
								string[] attributeInfo = inputLines[i + 1].Split(' ', StringSplitOptions.RemoveEmptyEntries); // The individual pieces of info.
								string[] attributeValues = new string[attributeInfo.Length - 1]; // The possible attribute values.
								Array.Copy(attributeInfo, 1, attributeValues, 0, attributeValues.Length);

								attributeList.Add(new Attribute(attributeInfo[0], attributeValues)); // Add the attribute to the list.
								attributeNames[i] = attributeInfo[0];

								// If the attribute is continuous, add it to the list of continuous attributes names and add its index to the list as well.
								if (attributeInfo[1].ToLower() == "continuous")
								{
									continuousAttributeNames.Add(attributeInfo[0]);
									continuousIndices.Add(i);
								}
								// If the attribute is to be ignored (kept for identification reasons only), then we add it to the ignore lists.
								else if (attributeInfo[1].ToLower() == "ignore")
								{
									ignoredAttributeNames.Add(attributeInfo[0]);
									ignoredIndices.Add(i);
								}
							}

							Console.WriteLine("Reading in classes...");
							// Gather possible classes.
							string[] classInfo = inputLines[attributeList.Count + 1].Split(' ', StringSplitOptions.RemoveEmptyEntries); // The class information.
							string[] classPossibilities = new string[classInfo.Length - 1]; // The possible classes for the tuples.
							Array.Copy(classInfo, 1, classPossibilities, 0, classPossibilities.Length);
							Attribute tupleClasses = new Attribute(classInfo[0], classPossibilities); // The tuple classes.

							// Gather the tuples.
							List<Tuple> tupleList = new List<Tuple>(); // The array of tuples in the training data.

							List<double>[] continuousValues = new List<double>[continuousAttributeNames.Count]; // The tuple values of each continuous attribute.
							for (int i = 0; i < continuousValues.Length; ++i)
							{
								continuousValues[i] = new List<double>();
							}

							Console.WriteLine("Reading in tuples...");
							// Read in the tuples.
							for (int i = 0; i < inputLines.Length - attributeList.Count - 2; ++i)
							{
								// The individual pieces of info.
								string[] tupleInfo = inputLines[i + 2 + attributeList.Count].Split(' ', StringSplitOptions.RemoveEmptyEntries);

								// Parse the values from the class. Note that the format is: attr1Val, attr2Val, ..., class.
								string[] tupleAttributes = new string[tupleInfo.Length - 1];
								Array.Copy(tupleInfo, 0, tupleAttributes, 0, tupleAttributes.Length);

								tupleList.Add(new Tuple(tupleInfo[^1], attributeNames, tupleAttributes));

								int continuousCount = 0; // The array index for our continuous attribute List array (from above).

								for (int j = 0; j < tupleAttributes.Length; ++j)
								{
									// For continuous attributes, we need to add the values to a special list.
									if (continuousIndices.Contains(j))
									{
										continuousValues[continuousCount].Add(Convert.ToDouble(tupleAttributes[j])); // Add the attribute value for the given attribute.
										++continuousCount;
									}
								}
							}

							// Add the continuous attributes to our dictionary object and sort them.
							for (int i = 0; i < continuousAttributeNames.Count; ++i)
							{
								continuousAttributeValues.Add(continuousAttributeNames[i], continuousValues[i]);
								continuousAttributeValues[continuousAttributeNames[i]].Sort();
							}

							Console.WriteLine("Discretizing continuous attributes...");
							// We need to deal with the continuous values by constraining them to two discrete options.
							foreach (string contName in continuousAttributeNames)
							{
								double bestSplitPoint = double.NaN; // The best split point based on info gain.

								double minExpectedInfo = double.MaxValue; // The minimum expected information for the partition.

								// Find the best split point.
								for (int i = 0; i < tupleList.Count - 1; ++i) // Check each split point.
								{
									// The split point is the midpoint between two continuous data points.
									double splitPoint = (continuousAttributeValues[contName][i] + continuousAttributeValues[contName][i + 1]) / 2.0;
									List<Tuple>[] subLists = new List<Tuple>[2]; // The two sublists for our trial partition.
									for (int j = 0; j < subLists.Length; ++j)
									{
										subLists[j] = new List<Tuple>();
									}

									// Populate the sublists.
									foreach (Tuple tupleBoi in tupleList)
									{
										if (Convert.ToDouble(tupleBoi.AttributeValues[contName]) <= splitPoint)
										{
											subLists[0].Add(tupleBoi);
										}
										else
										{
											subLists[1].Add(tupleBoi);
										}
									}

									double expectedInfo = ExpectedInfoWithPartition(tupleList, subLists, tupleClasses); // The expected info needed after making partition.

									// Update the expected info if necessary.
									if (expectedInfo < minExpectedInfo)
									{
										minExpectedInfo = expectedInfo;
										bestSplitPoint = splitPoint;
									}
								}

								// Adjust the continuous attribute to be discrete with two options: <=bestSplitPoint and >bestSplitPoint.
								for (int i = 0; i < tupleList.Count; ++i)
								{
									if (Convert.ToDouble(tupleList[i].AttributeValues[contName]) <= bestSplitPoint)
									{
										// "F1" specifies that we want 1 decimal value fixed notation.
										tupleList[i].AttributeValues[contName] = "<=" + bestSplitPoint.ToString("F2");
									}
									else
									{
										tupleList[i].AttributeValues[contName] = ">" + bestSplitPoint.ToString("F2");
									}
								}

								int contAttributeIndex = -1; // The index of the best attribute within the attributeList array.

								// The attribute list is index addressed, so we need to get the index associated with the name.
								for (int i = 0; i < attributeList.Count; ++i)
								{
									if (attributeList[i].Name == contName)
									{
										contAttributeIndex = i;
										break;
									}
								}

								// We now need to update the possible attribute values.
								attributeList[contAttributeIndex].Values = new string[2];
								attributeList[contAttributeIndex].Values[0] = "<=" + bestSplitPoint.ToString("F2");
								attributeList[contAttributeIndex].Values[1] = ">" + bestSplitPoint.ToString("F2");
							}

							Console.WriteLine("Building the decision tree (may take several minutes)...");
							Tree decisionTree = new Tree(string.Empty, string.Empty); // Our decision tree, built using recursion.

							// Perform the recursion needed to build the tree.
							TreeRecursion(tupleList, decisionTree.Root, attributeList, tupleClasses, continuousAttributeNames, ignoredAttributeNames);

							// Write the tree to file.
							Console.WriteLine("Writing tree to file...");
							string outputFileName = string.Empty; // The name of the output file.

							// Make sure we handle backslashes that may or may not be at the end of the directory.
							if (args[^1][^1] != '\\')
							{
								outputFileName = args[^1] + "\\" + Path.GetFileNameWithoutExtension(args[z]) + "_Classifier.txt";
							}
							else
							{
								outputFileName = args[^1] + Path.GetFileNameWithoutExtension(args[z]) + "_Classifier.txt";
							}

							// Create the directory if it does not already exist.
							Directory.CreateDirectory(Path.GetDirectoryName(outputFileName));

							StreamWriter outputFile = new StreamWriter(File.Create(outputFileName)); // Create the StreamWriter object to write to the file.
							WriteToFileDFS(decisionTree.Root, 0, outputFile);

							// Flush the buffer and close the file.
							outputFile.Flush();
							outputFile.Close();

							Console.WriteLine("Classifier written to " + outputFileName);
							Console.WriteLine();
						}
						catch (Exception ex)
						{
							// If something went wrong, the message will be displayed and the return value will be 1.
							Console.WriteLine("You done messed up, A-a-ron: " + ex.Message);
							return 1;
						}
					}
				}
				// We've built the classifier, now we want to use it.
				else if (args[0] == "-c")
                {
					try
					{
						string classifierFile = args[1];// This is where the classifier is located
						int numOfTupleFiles = args.Length - 4; //-2 for -c and <classiferfile>, -2 for <outputDirectory> and <default>

						Console.WriteLine("Classifying data based on decision tree in " + Path.GetFileName(classifierFile) + "...");
						string[] classifierLines = File.ReadAllLines(classifierFile); // The lines from the file.

						// Rebuild the tree based on the classifier file.
						Tree decisionTree = new Tree(string.Empty, string.Empty); // Our decision tree, to be reconstructed from the file.
						ReconstructTreeDFS(decisionTree.Root, classifierLines);

						for (int q = 0; q < numOfTupleFiles; ++q)
						{
							// Classify the tuples in each file.
							Console.WriteLine("Classifying tuples in input file " + Path.GetFileName(args[q + 2]) + "...");
							string currentInputTupleFile = args[q + 2]; // The current file containing tuples.
                            string[] currentInputFileLines = File.ReadAllLines(currentInputTupleFile);

							// First, we get the attributes present for each tuple.
							string[] attributeNames = currentInputFileLines[0].Split(' ', StringSplitOptions.RemoveEmptyEntries); // The attribute names.
							Tuple[] tuplesToClassify = new Tuple[currentInputFileLines.Length - 1];

							// Create the directory if it does not already exist.
							string outputFileName = string.Empty;

							// Make sure we handle backslashes that may or may not be at the end of the directory.
							if (args[^1][^1] != '\\')
							{
								outputFileName = args[^2] + "\\" + Path.GetFileNameWithoutExtension(args[q + 2]) + "_Classified.csv";
							}
							else
							{
								outputFileName = args[^2] + Path.GetFileNameWithoutExtension(args[q + 2]) + "_Classified.csv";
							}

							Directory.CreateDirectory(Path.GetDirectoryName(outputFileName));

							StreamWriter outputFile = new StreamWriter(File.Create(outputFileName)); // Create the StreamWriter object to write to the file.

							// Write the header with each attribute name, followed by Ans (for class).
							foreach (string attribute in attributeNames)
							{
								outputFile.Write(attribute + ",");
							}

							outputFile.WriteLine("Ans");

							// assuming first line is header
							for (int i = 1; i < currentInputFileLines.Length; ++i)
							{
								string[] attributeValues = currentInputFileLines[i].Split(' ', StringSplitOptions.RemoveEmptyEntries);
								tuplesToClassify[i - 1] = new Tuple(string.Empty, attributeNames, attributeValues);
								ClassifyTuples(decisionTree.Root, tuplesToClassify[i - 1]);// Stores the answer in tuplesToClassify[i - 1].Class.

								// If the tuple was not classified (did not have an exact match), then use the default.
								if (tuplesToClassify[i - 1].Class == string.Empty)
								{
									tuplesToClassify[i - 1].Class = args[^1];
								}
								
								// Write the classified tuple to file.
								foreach (string attribute in attributeNames)
								{
									outputFile.Write(tuplesToClassify[i - 1].AttributeValues[attribute] + ",");
								}
								outputFile.WriteLine(tuplesToClassify[i - 1].Class);
							}

							// Flush and close the output file.
							outputFile.Flush();
							outputFile.Close();

							Console.WriteLine("Classified data written to " + outputFileName);
							Console.WriteLine();
						}
					}
					catch (Exception ex)
					{
						// If something went wrong, the message will be displayed and the return value will be 1.
						Console.WriteLine("Looks like we ain't classifying after all. Here's what you did wrong: " + ex.Message);
						return 1;
					}
                }
			}
			return 0;
		}
	}
}