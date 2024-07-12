function [meanSensitivity, meanSpecificity, meanAccuracy, meanGmean]=wsvmmodel(data,a,b,w,t,mod)
% 3/14/2014 - Talayeh Razzaghi and Petros Xanthopoulos - Industral Engineering and Management Systems, University of central Florida - talayeh.razzaghi@gmail.com            
%                                                                                                
% INPUT:                                                                                                                                      
% data: time series data with label and weights, If we run SVM, the weights for all data samples are one                          
% a: the size of Normal class
% b: the size of abnormal class                           
% w: window length                                   
% t: parameter of abnormal pattern                             
% mod: if mod==1 SVM is running elseif mod==2 WSVM is running                                      
                                     
% OUTPUT:                                                                                          
% meanSensitivity: the average of sensitivity over 10-fold cross-validation
% meanSpecificity: the average of specificity over 10-fold cross-validation
% meanAccuracy: the average of Accuracy over 10-fold cross-validation
% meanGmean: the average of Gmean over 10-fold cross-validation

% Data Normalization
datanorm = zscore(data(:,1:w));       
data(:,1:w) = datanorm;                
classes = data(:,w+1);                 
Attributes = data(:,1:w);            
weight = data(:,w+2);                                    
ACC = [];
Sensitivity=[];
Specificity=[];
Gmean =[];

% 10-fold Cross-validation
for i = 1:10
    bb=0;
    cc=0;
    ROC=zeros(2,2);                  %contingency table
    r=randperm(numel(classes));
    tot=floor(numel(classes)*0.9);   % It selects 90% of data for training and 10% for testing      
    train=(r(1:tot));           %train=data(r(1:tot),:);
    train_l=classes(r(1:tot));
    
    test=(r(tot+3:end));            %test=(r(tot+1:end));            %test=data(r(tot+1:end),:);
    test_l=classes(r(tot+3:end));    %test_l=classes(r(tot+1:end));   %test lables
    % This avoids the test data to have empty set of abnormal data
    rarray = randperm(b)+a;
    rand1 = rarray(1);
    rand2 = rarray(2);
    test = [test rand1  rand2];
    test_l = [test_l; 0];
    test_l = [test_l; 0];
        
    if mod==1
        model = svmtrain(weight(train),classes(train),Attributes(train,:),'-s 0 -t 2 -c 10 -g 0.015618');      %SVM
    else
        model = svmtrain(weight(train),classes(train),Attributes(train,:),'-s 0 -t 2 -c 1000 -g 0.015618');    %WSVM
    end
    
    [predict_label , accuracy, prob_estimates] = svmpredict(classes(test), Attributes(test,:), model);
       
    % Accuracy
    acc=sum(predict_label==test_l)/size(test_l,1);
    fprintf('Accuracy on %d fold: %d\n', i, acc);
    ACC = [ACC; acc];
       
    % sensitivity & Specificity 
     for j=1:numel(test_l)
         if test_l(j)==1
            if predict_label(j)==1
                ROC(1,1)=ROC(1,1)+1;
            end
            if predict_label(j)==0
                ROC(2,1)=ROC(2,1)+1;
            end
        end
        
        if test_l(j)==0
            if predict_label(j)==1
                ROC(1,2)=ROC(1,2)+1;
            end
            if predict_label(j)==0
                ROC(2,2)=ROC(2,2)+1;
            end
        end
      end
        
    bb=ROC(1,1)+ROC(2,1);
    fprintf('b on %d fold: %d\n', i, b);
    sensitivity=ROC(1,1)/bb;
    fprintf('Sensitivity on %d fold: %d\n', i, sensitivity);
    Sensitivity = [Sensitivity; sensitivity];
    cc=ROC(2,2)+ROC(1,2);
    specificity=ROC(2,2)/cc;
    fprintf('specificity on %d fold: %d\n', i, specificity);
    Specificity = [Specificity;  specificity];
    gmean=sqrt(sensitivity*specificity);
    Gmean=[Gmean; gmean];    
end
% Mean accuracy, sensitivity, Specificity
% fprintf('Accuracy: %d, Stand dev: %d\n', mean(ACC), std(ACC));
% fprintf('Sensitivity: %d, Stand dev: %d\n', mean(Sensitivity), std(Sensitivity));
% fprintf('Specificity: %d, Stand dev: %d\n', mean(Specificity), std(Specificity))
meanSensitivity=mean(Sensitivity);
meanSpecificity=mean(Specificity);
meanAccuracy=mean(ACC);
meanGmean=mean(Gmean);

