#include <string>
#include <vector>
#include <fstream>
#include <cassert>
#include <iostream>
#include <cmath>

#include "classifier.h"
#include "EasyBMP.h"
#include "linear.h"
#include "argvparser.h"
#include "matrix.h"

using std::string;
using std::vector;
using std::ifstream;
using std::ofstream;
using std::pair;
using std::make_pair;
using std::cout;
using std::cerr;
using std::endl;

using CommandLineProcessing::ArgvParser;

typedef vector<pair<BMP*, int> > TDataSet;
typedef vector<pair<string, int> > TFileList;
typedef vector<pair<vector<float>, int> > TFeatures;

// Load list of files and its labels from 'data_file' and
// stores it in 'file_list'
void LoadFileList(const string& data_file, TFileList* file_list) {
    ifstream stream(data_file.c_str());

    string filename;
    int label;
    
    int char_idx = data_file.size() - 1;
    for (; char_idx >= 0; --char_idx)
        if (data_file[char_idx] == '/' || data_file[char_idx] == '\\')
            break;
    string data_path = data_file.substr(0,char_idx+1);
    
    while(!stream.eof() && !stream.fail()) {
        stream >> filename >> label;
        if (filename.size())
            file_list->push_back(make_pair(data_path + filename, label));
    }

    stream.close();
}

// Load images by list of files 'file_list' and store them in 'data_set'
void LoadImages(const TFileList& file_list, TDataSet* data_set) {
    for (size_t img_idx = 0; img_idx < file_list.size(); ++img_idx) {
            // Create image
        BMP* image = new BMP();
            // Read image from file
        image->ReadFromFile(file_list[img_idx].first.c_str());
            // Add image and it's label to dataset
        data_set->push_back(make_pair(image, file_list[img_idx].second));
    }
}

// Save result of prediction to file
void SavePredictions(const TFileList& file_list,
                     const TLabels& labels, 
                     const string& prediction_file) {
        // Check that list of files and list of labels has equal size 
    assert(file_list.size() == labels.size());
        // Open 'prediction_file' for writing
    ofstream stream(prediction_file.c_str());

        // Write file names and labels to stream
    for (size_t image_idx = 0; image_idx < file_list.size(); ++image_idx)
        stream << file_list[image_idx].first << " " << labels[image_idx] << endl;
    stream.close();
}

// Exatract features from dataset.
// You should implement this function by yourself =)

void GrayScale (BMP* Image, Matrix<double> ImageMatrix) {
	for (int i = 0; i < Image->TellHeight(); i++)
		for (int j = 0; j < Image->TellWidth(); j++) {
			RGBApixel P = Image->GetPixel(j, i);
			ImageMatrix(i, j) = 0.299*P.Red + 0.587*P.Green + 0.114*P.Blue;
		}
}

void SobelX (Matrix<double> ImageMatrix, Matrix<double> ImageMatrix1) {
	//int SobelVector[3] = {-1, 0, 1};
	for (uint i = 0; i < ImageMatrix.n_rows; i++) {
		for (uint j = 0; j < ImageMatrix.n_cols - 2; j++) {
			ImageMatrix1(i, j) = ImageMatrix(i, j+2)-ImageMatrix(i,j);
		}
	}
}

void SobelY (Matrix<double> ImageMatrix, Matrix<double> ImageMatrix1) {
	//int SobelVector[3] = {1, 0, -1};
	for (uint i = 0; i < ImageMatrix.n_rows - 2; i++) {
		for (uint j = 0; j < ImageMatrix.n_cols; j++) {
			ImageMatrix1(i, j) = ImageMatrix(i,j) - ImageMatrix(i + 2, j);
		}
	}
}

void Grad (Matrix<double> ImageMatrix1, Matrix<double> ImageMatrix2, 
           Matrix<double> IG, Matrix<double> IGO) {
	for (uint i = 0; i < ImageMatrix1.n_rows; i++)
		for (uint j = 0; j < ImageMatrix1.n_cols; j++){
			IG(i, j) = sqrt(ImageMatrix1(i, j)*ImageMatrix1(i, j)+ImageMatrix2(i, j)*ImageMatrix2(i, j)); //модуль градиента
			IGO(i, j) = atan2(ImageMatrix1(i, j), ImageMatrix2(i, j)); //направление градиента
		}
}

void Histogram (Matrix <double> Block.Orient, Matrix <double> Block.Modul){
    struct Hist { //структура для гистограммы
        double Orient[8]; //значения арктангенса на краях интервала
        double Modul; //заведём массив, в котором будут модули градиента для этого промежутка (Дописать размер)
    };
    Hist Step;
    
    const double c = 3.14*2/8; //от -3.14 до 3.14 делим на 8 сегментов
    bool flag; //флаг для опеделения попадания в сегмент
    
    Step.Orient[0] = -3.14; //определим значения на краях сегментов
    for (uint q = 1; q < 8; q++)
        Step.Orient[q] = Step.Orient[q-1]+c;
    
    for (uint i = 0; ??; i++)
        for (uint j = 0; ??; j++) {
            flag = true
            q = 0;
            do {
                if ((Block.Orient(i, j) >= Step.Orient[q])&&(Block.Orient(i, j) <= Step.Orient[q+1])) { //если направление в блоке попало в сегмент
                    Step.Modul[q] += Block.Modul(i, j); //увеличим значение элемента гистограммы, где хранятся модули градиента
                    flag = false; // и перейдём к следующему пикселю
                }
                else q++; //или смотрим следующий сегмент
            } while ((flag)||(q==8)); //занимаемся этим, пока не попадём в сегмент или не проверим все сегменты
        }
    
}

void Division (Matrix <double> ImageMatrix) {
    struct SBlock { //создаём структуру для клетки, каждому пикселю соответствует
        Matrix <double> Orient; //направление (Дописать размер)
        Matrix <double> Modul; //значение модуля (Дописать размер)
    };
    
    SBlock Block[64]; //сама клетка. делить изображение будем на 8х8=64 клетки. 
                      //Имеем массив из этих структур, чтобы гистограммы хранились для каждой клетки.
    for (uint k = 0; k < 64; k++){
        for (uint i = ImageMatrix.n_rows*k/8, m = 0; i < ImageMatrix.n_rows*(k+1)/8; i++, m++)
            for (uint j = ImageMatrix.n_cols*k/8, n = 0; j < ImageMatrix.n_cols*(k+1)/8; j++, n++){
                Block[k].Orient(m, n) = IGO(i, j);
                Block[k].Modul(m, n) = IG(i, j);
            }
        Histogram(Block[k].Orient, Block[k].Modul); //строим гистограмму для клетки
    }
}

void ExtractFeatures(const TDataSet& data_set, TFeatures* features) {
    for (size_t image_idx = 0; image_idx < data_set.size(); ++image_idx) {
        Matrix <double> ImageMatrix  (data_set[image_idx].first->TellHeight(), data_set[image_idx].first->TellWidth());
        Matrix <double> ImageMatrix1 (ImageMatrix.n_rows, ImageMatrix.n_cols);
        Matrix <double> ImageMatrix2 (ImageMatrix.n_rows, ImageMatrix.n_cols);
        Matrix <double> ImageGradient (ImageMatrix.n_rows, ImageMatrix.n_cols);
        Matrix <double> ImageGradientOrientation (ImageMatrix.n_rows, ImageMatrix.n_cols);
	//cout << data_set[image_idx].first->TellHeight()<< " " << data_set[image_idx].first->TellWidth()<< endl;	
        GrayScale(data_set[image_idx].first, ImageMatrix);
        SobelX(ImageMatrix, ImageMatrix1);
        SobelY(ImageMatrix, ImageMatrix2);
        Grad(ImageMatrix1, ImageMatrix2, ImageGradient, ImageGradientOrientation);
        
        // PLACE YOUR CODE HERE
        // Remove this sample code and place your feature extraction code here
        vector<float> one_image_features;
        one_image_features.push_back(1.0);
        features->push_back(make_pair(one_image_features, 1));
        // End of sample code

    }
}

// Clear dataset structure
void ClearDataset(TDataSet* data_set) {
        // Delete all images from dataset
    for (size_t image_idx = 0; image_idx < data_set->size(); ++image_idx)
        delete (*data_set)[image_idx].first;
        // Clear dataset
    data_set->clear();
}

// Train SVM classifier using data from 'data_file' and save trained model
// to 'model_file'
void TrainClassifier(const string& data_file, const string& model_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // Model which would be trained
    TModel model;
        // Parameters of classifier
    TClassifierParams params;
    
        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // PLACE YOUR CODE HERE
        // You can change parameters of classifier here
    params.C = 0.01;
    TClassifier classifier(params);
        // Train classifier
    classifier.Train(features, &model);
        // Save model to file
    model.Save(model_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

// Predict data from 'data_file' using model from 'model_file' and
// save predictions to 'prediction_file'
void PredictData(const string& data_file,
                 const string& model_file,
                 const string& prediction_file) {
        // List of image file names and its labels
    TFileList file_list;
        // Structure of images and its labels
    TDataSet data_set;
        // Structure of features of images and its labels
    TFeatures features;
        // List of image labels
    TLabels labels;

        // Load list of image file names and its labels
    LoadFileList(data_file, &file_list);
        // Load images
    LoadImages(file_list, &data_set);
        // Extract features from images
    ExtractFeatures(data_set, &features);

        // Classifier 
    TClassifier classifier = TClassifier(TClassifierParams());
        // Trained model
    TModel model;
        // Load model from file
    model.Load(model_file);
        // Predict images by its features using 'model' and store predictions
        // to 'labels'
    classifier.Predict(features, model, &labels);

        // Save predictions
    SavePredictions(file_list, labels, prediction_file);
        // Clear dataset structure
    ClearDataset(&data_set);
}

int main(int argc, char** argv) {
    // Command line options parser
    ArgvParser cmd;
        // Description of program
    cmd.setIntroductoryDescription("Machine graphics course, task 2. CMC MSU, 2014.");
        // Add help option
    cmd.setHelpOption("h", "help", "Print this help message");
        // Add other options
    cmd.defineOption("data_set", "File with dataset",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("model", "Path to file to save or load model",
        ArgvParser::OptionRequiresValue | ArgvParser::OptionRequired);
    cmd.defineOption("predicted_labels", "Path to file to save prediction results",
        ArgvParser::OptionRequiresValue);
    cmd.defineOption("train", "Train classifier");
    cmd.defineOption("predict", "Predict dataset");
        
        // Add options aliases
    cmd.defineOptionAlternative("data_set", "d");
    cmd.defineOptionAlternative("model", "m");
    cmd.defineOptionAlternative("predicted_labels", "l");
    cmd.defineOptionAlternative("train", "t");
    cmd.defineOptionAlternative("predict", "p");

        // Parse options
    int result = cmd.parse(argc, argv);

        // Check for errors or help option
    if (result) {
        cout << cmd.parseErrorDescription(result) << endl;
        return result;
    }

        // Get values 
    string data_file = cmd.optionValue("data_set");
    string model_file = cmd.optionValue("model");
    bool train = cmd.foundOption("train");
    bool predict = cmd.foundOption("predict");

        // If we need to train classifier
    if (train)
        TrainClassifier(data_file, model_file);
        // If we need to predict data
    if (predict) {
            // You must declare file to save images
        if (!cmd.foundOption("predicted_labels")) {
            cerr << "Error! Option --predicted_labels not found!" << endl;
            return 1;
        }
            // File to save predictions
        string prediction_file = cmd.optionValue("predicted_labels");
            // Predict data
        PredictData(data_file, model_file, prediction_file);
    }
}