#include <iostream>
#include <stdlib.h>
#include <vector>
#include <sstream>
#include <fstream>
#include <limits>

using namespace std;

//This struct stores the class label of an instance as well as a list of all the values of the features
//"feature 1" would be the first value in the list for all nodes, and all nodes should have same size list
struct node {
	int cLabel = 0;
	vector<double> features;
};

//Reads the input data from a file and stores it in a list of all instances
void readData(vector<node*>&, string);

//conducts a forward selection greedy search on a list of instances, passes in the number of features for ease
void forwardFeatureSearch(vector<node*>, int);

//conducts a backward elimination greedy search on a list of instances, passes in the number of features for ease
void backwardFeatureSearch(vector<node*>, int);

//checks if a certain int is in the given list of ints (used to check if we already have a feature in the feature list)
bool contains(vector<int>, int);

//using the list of instances and a given list of features to look at, it performs k-fold cross validatation
//with k = 1 using only the features already selected features + the first passed in int as a potential feature
//the second int distinguishes whether the first int should be removed or added to the passed in feature list
double crossValid(vector<node*>, vector<int>, int, int);

//finds the euclidian distance bewteen two instances using the list of features passed in
// d=sqrt((I1f1-I2f1)^2 + (I1f2-I2f2)^2 + ... + (I1fn-I2fn)^2)
double findDistance(node*, node*, vector<int>);

int main() {
	vector<node*> data;   //stores all instances, will be filled by readData
	string filename;   //stores name of the input file, to be recieved from user input
	int numFeatures = 0; //stores the total number of features
	int algorithmChoice = 0; //stores user's choice of algorithm

	//recieving file name
	cout << "Welcome to Thomas Henningson's Feature Selection Algorithm.\n";
	cout << "Type in the name of the file to test: ";
	cin >> filename;

	readData(data, filename);   //fills the data vector from file

	//something went wrong, no data or no file
	if (data.empty()) {
		cout << "Could not find file or data set empty. Exiting...\n";
		return 1;
	}

	numFeatures = data.at(0)->features.size();   //#of features = size of any node's feature list

	//edge case where user decides no features should be inputted
	if (numFeatures == 0) {
		cout << "Data set contains no features. Exiting...\n";
		return 1;
	}

	//recieving algorithm choice
	cout << "\nType the number of the algorithm you want to run.\n";
	cout << "   1) Forward Selection\n   2) Backward Elimination\n";
	cin >> algorithmChoice;
	while (algorithmChoice != 1 && algorithmChoice != 2) {
		cout << "\nPlease select 1 or 2\n";
		cout << "   1) Forward Selection\n   2) Backward Elimination\n";
		cin >> algorithmChoice;
	}

	//extra information to be given to user
	cout << "\nThis dataset has " << numFeatures << " features(not including the class attribute), with " << data.size() << " instances.\n";
	cout << endl;

	//run selected algorithm
	if (algorithmChoice == 1) {
		forwardFeatureSearch(data, numFeatures);
	}
	else {;
		backwardFeatureSearch(data, numFeatures);
	}

	return 0;
}

//Reads the input data from a file and stores it in a list of all instances
void readData(vector<node*>& v, string filename) {
	stringstream ss;   //used to gather discrete numbers from a string
	string str;   //used to hold a single line from the file
	ifstream IF;   //used to open the file
	node* temp;   //used to hold an instance before it is added to the list
	double num;   //used to gather numbers from the stringstream

	//open file, exit if it cannot
	IF.open(filename);
	if (!IF.is_open()) {
		return;
	}

	//runs until all lines in the file have been proccesed (each line is a seperate instance)
	while (getline(IF, str)) {
		temp = new node;   //create new instance
		ss << str;   //convert to stringstream
		ss >> num;   //class label
		temp->cLabel = num;   //store class label

		//runs until all feature data is processed
		while (ss>>num) {
			//each num is value of feature 1, then 2, and so on, so pushback maintains correct ordering
			temp->features.push_back(num);
		}
		ss.clear();   //clear stringstream for next line
		v.push_back(temp);   //add instance to the list
	}
	IF.close();   //done with file, so close it
}

//conducts a forward selection greedy search on a list of instances, passes in the number of features for ease
void forwardFeatureSearch(vector<node*> data, int numFeatures) {
	vector<int> currentFeatureSet;   //stores the greedy feature choice at each level
	vector<int> bestFeatures;   //stores the list of features with the highest current accuracy
	int featureToAdd;   //used to find feature with best accuracy at each level
	double accuracy;   //holds current feature list accuracy at each level
	double bestAccuracy;   //holds best feature list accuracy at each level
	double bestTotalAccuracy = 0;   //holds the absolute best feature list accuracy

	cout << "Beginning search.\n\n";
	//will add one feature at each level, so loop for that many levels
	for (int i = 0; i < numFeatures; i++) {
		bestAccuracy = 0;   //reset best accuracy for this level
		//must check all features potential at each level
		for (int j = 0; j < numFeatures; j++) {
			//checks if feature is already in the list from a previous level
			if (!contains(currentFeatureSet, j + 1)) {
				accuracy = crossValid(data, currentFeatureSet, j+1, 1);   //find accuracy with list + potential new feature
				//user feedback
				cout << "   Using feature(s) {" << j+1;
				for (int k = 0; k < currentFeatureSet.size(); k++) {
					cout << "," << currentFeatureSet.at(k);
				}
				cout << "} accuracy is " << accuracy*100 << "%\n";
				//if this new potential feature has a better accuracy at this level, replace old potential feature
				if (accuracy > bestAccuracy) {
					bestAccuracy = accuracy;
					featureToAdd = j + 1;
				}
			}
		}
		//found best potential feature to add to the list at this level, so add it
		currentFeatureSet.push_back(featureToAdd);
		cout << endl;
		//if this new feature also introduces a list with a better overall accuracy at all levels, then store it
		if (bestAccuracy > bestTotalAccuracy) {
			bestTotalAccuracy = bestAccuracy;
			bestFeatures = currentFeatureSet;
		}
		else if (bestAccuracy < bestTotalAccuracy) {
			cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n";
		}
		else {}
		//user feedback
		cout << "Feature set {";
		for (int k = 0; k < currentFeatureSet.size(); k++) {
			if (k < currentFeatureSet.size() - 1) {
				cout << currentFeatureSet.at(k) << ",";
			}
			else {
				cout << currentFeatureSet.at(k);
			}
		}
		cout << "} was best, accuracy is " << bestAccuracy*100 << "%\n\n";
	}
	//finished search, output best feature list algorithm found, and its accuracy
	cout << "Finished search!! The best feature subset is {";
	for (int k = 0; k < bestFeatures.size(); k++) {
		if (k < bestFeatures.size() - 1) {
			cout << bestFeatures.at(k) << ",";
		}
		else {
			cout << bestFeatures.at(k);
		}
	}
	cout << "}, which has an accuracy of " << bestTotalAccuracy*100 << "%\n";
}

//conducts a backward elimination greedy search on a list of instances, passes in the number of features for ease
void backwardFeatureSearch(vector<node*> data, int numFeatures) {
	vector<int> currentFeatureSet;   //stores current features being looked at, starts with all features
	vector<int> bestFeatures;   //stores feature list with highest accuracy at all levels
	int featureToRemove = -1;   //potential feature to eliminate from list
	double accuracy;   //holds current feature list accuracy at each level
	double bestAccuracy;   //holds best feature list accuracy at each level
	double bestTotalAccuracy = 0;   //holds the absolute best feature list accuracy
	int index;   //holds the index location in the list of the feature to remove

	cout << "Beginning search.\n\n";
	//start with all features
	for (int i = 0; i < numFeatures; i++) {
		currentFeatureSet.push_back(i + 1);
	}
	accuracy = crossValid(data, currentFeatureSet, -1, -1);   //special case where we have to check all features in case it is best
	//user feedback
	cout << "   Using feature(s) {";
	for (int k = 0; k < currentFeatureSet.size(); k++) {
		if (k!= currentFeatureSet.size()-1) {
			cout << currentFeatureSet.at(k) << ",";
		}
		else {
			cout << currentFeatureSet.at(k);
		}
	}
	cout << "} accuracy is " << accuracy * 100 << "%\n";
	cout << "\nFeature set {";
	for (int k = 0; k < currentFeatureSet.size(); k++) {
		if (k < currentFeatureSet.size() - 1) {
			cout << currentFeatureSet.at(k) << ",";
		}
		else {
			cout << currentFeatureSet.at(k);
		}
	}
	cout << "} was best, accuracy is " << accuracy * 100 << "%\n\n";

	//current best features and accuracy are the special case tested
	bestTotalAccuracy = accuracy;
	bestFeatures = currentFeatureSet;

	//run the other n-1 levels (not the first level which we already ran)
	for (int i = 0; i < numFeatures-1; i++) {
		bestAccuracy = 0;   //reset best accuracy for this level
		//must check all features potential to remove at each level
		for (int j = 0; j < numFeatures; j++) {
			//checks if feature has already been removed
			if (contains(currentFeatureSet, j + 1)) {
				accuracy = crossValid(data, currentFeatureSet, j + 1, 2);   //find accuracy with list - potential feature to eliminate
				//user feedback
				cout << "   Using feature(s) {";
				if (currentFeatureSet.size() >= 2) {
					if (currentFeatureSet.at(0) == j+1) {
						cout << currentFeatureSet.at(1);
					}
					else if(currentFeatureSet.at(1) == j+1) {
						cout << currentFeatureSet.at(0);
					}
					else {
						cout << currentFeatureSet.at(0) << "," << currentFeatureSet.at(1);
					}
				}
				else if (!currentFeatureSet.empty()) {
					cout << currentFeatureSet.at(0);
				}
				else {}
				for (int k = 2; k < currentFeatureSet.size(); k++) {
					 if (currentFeatureSet.at(k) != j + 1) {
						cout << "," << currentFeatureSet.at(k);
					 }
				}
				cout << "} accuracy is " << accuracy * 100 << "%\n";

				//if accuracy increased or stayed the same for the removed feature at this level, store it
				if (accuracy >= bestAccuracy) {
					bestAccuracy = accuracy;
					featureToRemove = j + 1;
				}
			}
		}
		//find index of feature to remove
		index = currentFeatureSet.size() - 1;
		for (int j = 0; j < currentFeatureSet.size(); j++) {
			if (currentFeatureSet.at(j) == featureToRemove) {
				index = j;
			}
		}
		//swap with last feature in the list and pop_back
		currentFeatureSet.at(index) = currentFeatureSet.at(currentFeatureSet.size() - 1);
		currentFeatureSet.pop_back();
		cout << endl;
		//if the feature list at this level has a better or equal accuracy the other levels, store it
		if (bestAccuracy >= bestTotalAccuracy) {
			bestTotalAccuracy = bestAccuracy;
			bestFeatures = currentFeatureSet;
		}
		else if (bestAccuracy < bestTotalAccuracy) {
			cout << "(Warning, Accuracy has decreased! Continuing search in case of local maxima)\n";
		}
		else {}
		//user feedback
		cout << "Feature set {";
		for (int k = 0; k < currentFeatureSet.size(); k++) {
			if (k < currentFeatureSet.size() - 1) {
				cout << currentFeatureSet.at(k) << ",";
			}
			else {
				cout << currentFeatureSet.at(k);
			}
		}
		cout << "} was best, accuracy is " << bestAccuracy * 100 << "%\n\n";
	}
	//finished search, output best feature list algorithm found, and its accuracy
	cout << "Finished search!! The best feature subset is {";
	for (int k = 0; k < bestFeatures.size(); k++) {
		if (k < bestFeatures.size() - 1) {
			cout << bestFeatures.at(k) << ",";
		}
		else {
			cout << bestFeatures.at(k);
		}
	}
	cout << "}, which has an accuracy of " << bestTotalAccuracy * 100 << "%\n";
}

//checks if a certain int is in the given list of ints (used to check if we already have a feature in the feature list)
bool contains(vector<int> v, int num) {
	for (int i = 0; i < v.size(); i++) {
		if (v.at(i) == num) {
			return true;
		}
	}
	return false;
}

//using the list of instances and a given list of features to look at, it performs k-fold cross validatation using nearest neighbor
//with k = 1 using only the features already selected features + the first passed in int as a potential feature
//the second int distinguishes whether the first int should be removed or added to the passed in feature list
double crossValid(vector<node*> data, vector<int> currentFeatureSet, int feature, int choice) {
	//pushback potential feature if forward selection
	if (choice == 1) {
		currentFeatureSet.push_back(feature);
	}
	//remove potential feature if backward elimination
	else if(choice == 2) {
		int index = currentFeatureSet.size() - 1;
		for (int j = 0; j < currentFeatureSet.size(); j++) {
			if (currentFeatureSet.at(j) == feature) {
				index = j;
			}
		}
		currentFeatureSet.at(index) = currentFeatureSet.at(currentFeatureSet.size() - 1);
		currentFeatureSet.pop_back();
	}
	else {}
	int correctlyClassified = 0;   //holds number of correct classifications(tested using nearest neighbor)
	double nnDist;   //holds distance of nearest beighbor
	int nnLocation;   //holds index of nearest neighbor in list of instances
	double distance;   //holds a distance from one instance to another
	node* objectToClassify;   //holds the instance to be classified in the test, using all other instances
	for (int i = 0; i < data.size(); i++) {
		objectToClassify = data.at(i);   //will classify all instances using all other instances at each loop
		nnDist = numeric_limits<double>::max();   //reset nearest neighbors distance with infinty
		nnLocation = -1;   //reset nearest neighbor location
		//loop through all instances to find nearest neighbor
		for (int j = 0; j < data.size(); j++) {
			//skip instance we are attempting to validate/classify
			if (i != j) {
				distance = findDistance(data.at(j), objectToClassify, currentFeatureSet); //find distance between this instance and the validation instance, using only the features we are looking at
				//if distance is smaller then current nearest neighbor, this instance is our new nearest neighbor
				if (distance < nnDist) {
					nnDist = distance;
					nnLocation = j;
				}
			}
		}
		//check if using the same label as nearest neighbor would have correctly classified this instance
		if (data.at(nnLocation)->cLabel == objectToClassify->cLabel) {
			correctlyClassified++;   //if yes, then increase the #correctly classified
		}
	}
	//accuracy is equal to #correct/total number tested
	return ((double)(correctlyClassified) / data.size());
}

//finds the euclidian distance bewteen two instances using the list of features passed in
// d=sqrt((I1f1-I2f1)^2 + (I1f2-I2f2)^2 + ... + (I1fn-I2fn)^2)
double findDistance(node* objectToCompare, node* objectToClassify, vector<int> currentFeatureSet) {
	double sum = 0;   //holds sum of the all the differences of feature values squared
	double tempValue = 0;   //used for sum calculation
	//loop through all features being looked at
	for (int i = 0; i < currentFeatureSet.size(); i++) {
		//find differece in feature value
		tempValue = (objectToCompare->features.at(currentFeatureSet.at(i) - 1) - objectToClassify->features.at(currentFeatureSet.at(i) - 1));
		tempValue = tempValue * tempValue;   //square difference
		sum = tempValue + sum;   //ass it to sum
	}
	return sqrt(sum); //distance is the sqrt of the sum of differences
}