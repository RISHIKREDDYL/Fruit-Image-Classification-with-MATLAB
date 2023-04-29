% Load image data
Dataset = imageDatastore('C:\Users\Rishik Reddy\Downloads\Fruit-Classification_Kaggle\train\train\', ...
    'IncludeSubfolders', true, 'LabelSource', 'foldernames');

% Reducing all labels to 200 images
trainingDatastore = splitEachLabel(Dataset, 200, 'randomize');
countEachLabel(trainingDatastore)

% Split dataset between training, validation and testing sets
[Training_Dataset, Validation_Dataset, Test_Dataset] = splitEachLabel(trainingDatastore, 0.7, 0.2, 0.1);
countEachLabel(Training_Dataset)
countEachLabel(Validation_Dataset)
countEachLabel(Test_Dataset)

% Load pre-trained GoogleNet
net = googlenet;

% Resize images to match input layer size
Input_Layer_Size =  net.Layers(1).InputSize(1:2);
Resized_Training_Images = augmentedImageDatastore(Input_Layer_Size, Training_Dataset);
Resized_Validation_Images = augmentedImageDatastore(Input_Layer_Size, Validation_Dataset);
Resized_Testing_Images = augmentedImageDatastore(Input_Layer_Size, Test_Dataset);

% Replace last layers for classification task
Feature_Learner = net.Layers(142);
Output_Classifier = net.Layers(144);
Number_of_Classes = numel(categories(Training_Dataset.Labels));

New_Feature_Learner = fullyConnectedLayer(Number_of_Classes, ...
    'Name', 'Fruit and Vegetable Feature Learner');
New_Classifier_Layer = classificationLayer('Name', 'Fruit and Vegetable Classifier');

Layer_Graph = layerGraph(net);
New_Layer_Graph = replaceLayer(Layer_Graph, Feature_Learner.Name, New_Feature_Learner);
New_Layer_Graph = replaceLayer(New_Layer_Graph, Output_Classifier.Name, New_Classifier_Layer);

% Set up training options
Training_Options = trainingOptions('sgdm', ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.01,...
    'LearnRateDropPeriod',10,...
    'MiniBatchSize', 46, ...
    'MaxEpochs', 5, ...
    'InitialLearnRate', 0.01, ...
    'Shuffle', 'every-epoch', ...
    'ValidationData', Resized_Validation_Images, ...
    'ValidationFrequency', 50, ...
    'Verbose', false, ...
    'Plots', 'training-progress');

% Train network and save the training progress
[net,info] = trainNetwork(Resized_Training_Images, New_Layer_Graph, Training_Options);
save net
save training_info info

% Classify the test set
predictedLabels = classify(net, Resized_Testing_Images);

% Get the actual labels for the test set
actualLabels = Test_Dataset.Labels;

% Create a confusion matrix
confusionMatrix = confusionchart(actualLabels, predictedLabels);

% Calculate the accuracy
accuracy = mean(predictedLabels == actualLabels);

% Display the accuracy
disp(['Accuracy on the test set: ', num2str(accuracy)]);