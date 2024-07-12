function [meanACC, meanROC]=wsvmmodelmulti(data,w,n,m,mod)  
% 3/14/2014 - Talayeh Razzaghi and Petros Xanthopoulos - Industral Engineering and Management Systems, University of central Florida - talayeh.razzaghi@gmail.com            
%                                                                                                
% INPUT:                                                                                                                                      
% data: time series data with label and weights, If we run SVM, the weights for all data samples are one                          
% m: the size of Normal class
% n: the size of abnormal class                          
% w: window length                                   
% t: parameter of abnormal pattern                             
% mod: if mod==1 SVM is running elseif mod==2 WSVM is running                                      
                                     
% OUTPUT:                                                                                          
% meanROC: the average of ROC over 10-fold cross-validation
% meanACC: the average of Accuracy over 10-fold cross-validation

% Data Normalization
datanorm = zscore(data(:,1:w));         
data(:,1:w) = datanorm;                 
classes = data(:,w+1);                    
weight = data(:,w+2);                 

train_l=[];
test_l=[];

ROC2=zeros(8,8);
ACC=[];
sos1=0;    sos2=0;    sos3=0;    sos4=0;    sos5=0;    sos6=0;    sos7=0;    sos8=0;

% 10-fold Cross-validation
for i = 1:10
    b1=0;
    b2=0;
    b3=0;
    b4=0;
    b5=0;
    b6=0;
    b7=0;
    
    ROC=zeros(8,8);    %contingency table
    ROC1=zeros(8,8);
    
    r = randperm(numel(classes));
    tot = floor(numel(classes)*0.9);
    train = (r(1:tot));           
    train_l = classes(r(1:tot));
    traindata = data(r(1:tot),:);
    trainweight = weight(r(1:tot),:);
    
    test = (r(tot+8:end));        %test=data(r(tot+1:end),:);
    test_l = classes(r(tot+8:end));
    testdata = data(r(tot+8:end),:);
    
    % This avoids the test data to have empty set of abnormal data
    rarray1 = randperm(n) + m;
    rand1 = rarray1(1);
    rarray2 = randperm(n) + n + m;
    rand2 = rarray2(1);
    rarray3 = randperm(n) + 2*n+ m;
    rand3 = rarray3(1);
    rarray4 = randperm(n) + 3*n + m;
    rand4 = rarray4(1);
    rarray5 = randperm(n) + 4*n + m;
    rand5 = rarray5(1);
    rarray6 = randperm(n) + 5*n + m;
    rand6 = rarray6(1);
    rarray7 = randperm(n) + 6*n + m;
    rand7 = rarray7(1);
      
    test_l = [test_l; 2; 3; 4; 5; 6; 7; 8];
    testdata = [testdata; data(rand1,:); data(rand2,:); data(rand3,:); data(rand4,:); data(rand5,:); data(rand6,:); data(rand7,:)]; 
    
    if mod==1
        model = svmtrain(trainweight,train_l, traindata,'-s 0 -t 2 -c 10 -g 0.015618');    %c 1000 WSVM               %c 10 SVM
    else
        model = svmtrain(trainweight,train_l, traindata,'-s 0 -t 2 -c 1000 -g 0.015618');    %c 1000 WSVM               %c 10 SVM
    end
    
    [predict_label , accuracy, prob_estimates] = svmpredict(test_l, testdata, model);
        
    % Accuracy
    acc = sum(predict_label==test_l)/size(test_l,1);
    fprintf('Accuracy on %d fold: %d\n', i, acc);
    ACC = [ACC; acc];
       
   % Confusion Matrix 
     for j=1:numel(test_l)
         if test_l(j)==1
            if predict_label(j)==1
                ROC(1,1)=ROC(1,1)+1;
            end
            if predict_label(j)==2
                ROC(2,1)=ROC(2,1)+1;
            end
            if predict_label(j)==3
                ROC(3,1)=ROC(3,1)+1;
            end
            if predict_label(j)==4
                ROC(4,1)=ROC(4,1)+1;
            end
            if predict_label(j)==5
                ROC(5,1)=ROC(5,1)+1;
            end
            if predict_label(j)==6
                ROC(6,1)=ROC(6,1)+1;
            end
            if predict_label(j)==7
                ROC(7,1)=ROC(7,1)+1;
            end
            if predict_label(j)==8
                ROC(8,1)=ROC(8,1)+1;
            end
         end
        
        if test_l(j)==2
            if predict_label(j)==1
                ROC(1,2)=ROC(1,2)+1;
            end
            if predict_label(j)==2
                ROC(2,2)=ROC(2,2)+1;
            end
            if predict_label(j)==3
                ROC(3,2)=ROC(3,2)+1;
            end
            if predict_label(j)==4
                ROC(4,2)=ROC(4,2)+1;
            end
            if predict_label(j)==5
                ROC(5,2)=ROC(5,2)+1;
            end
            if predict_label(j)==6
                ROC(6,2)=ROC(6,2)+1;
            end
            if predict_label(j)==7
                ROC(7,2)=ROC(7,2)+1;
            end
            if predict_label(j)==8
                ROC(8,2)=ROC(8,2)+1;
            end
        end
        
         if test_l(j)==3
            if predict_label(j)==1
                ROC(1,3)=ROC(1,3)+1;
            end
            if predict_label(j)==2
                ROC(3,2)=ROC(3,2)+1;
            end
            if predict_label(j)==3
                ROC(3,3)=ROC(3,3)+1;
            end
            if predict_label(j)==4
                ROC(4,3)=ROC(4,3)+1;
            end
            if predict_label(j)==5
                ROC(5,3)=ROC(5,3)+1;
            end
            if predict_label(j)==6
                ROC(6,3)=ROC(6,3)+1;
            end
            if predict_label(j)==7
                ROC(7,3)=ROC(7,3)+1;
            end
            if predict_label(j)==8
                ROC(8,3)=ROC(8,3)+1;
            end
         end
         
         if test_l(j)==4
            if predict_label(j)==1
                ROC(1,4)=ROC(1,4)+1;
            end
            if predict_label(j)==2
                ROC(2,4)=ROC(2,4)+1;
            end
            if predict_label(j)==3
                ROC(3,4)=ROC(3,4)+1;
            end
            if predict_label(j)==4
                ROC(4,4)=ROC(4,4)+1;
            end
            if predict_label(j)==5
                ROC(5,4)=ROC(5,4)+1;
            end
            if predict_label(j)==6
                ROC(6,4)=ROC(6,4)+1;
            end
            if predict_label(j)==7
                ROC(7,4)=ROC(7,4)+1;
            end
            if predict_label(j)==8
                ROC(8,4)=ROC(8,4)+1;
            end
         end
         
         if test_l(j)==5
            if predict_label(j)==1
                ROC(1,5)=ROC(1,5)+1;
            end
            if predict_label(j)==2
                ROC(2,5)=ROC(2,5)+1;
            end
            if predict_label(j)==3
                ROC(3,5)=ROC(3,5)+1;
            end
            if predict_label(j)==4
                ROC(4,5)=ROC(4,5)+1;
            end
            if predict_label(j)==5
                ROC(5,5)=ROC(5,5)+1;
            end
            if predict_label(j)==6
                ROC(6,5)=ROC(6,5)+1;
            end
            if predict_label(j)==7
                ROC(7,5)=ROC(7,5)+1;
            end
            if predict_label(j)==8
                ROC(8,5)=ROC(8,5)+1;
            end
         end
         
         if test_l(j)==6
            if predict_label(j)==1
                ROC(1,6)=ROC(1,6)+1;
            end
            if predict_label(j)==2
                ROC(2,6)=ROC(2,6)+1;
            end
            if predict_label(j)==3
                ROC(3,6)=ROC(3,6)+1;
            end
            if predict_label(j)==4
                ROC(4,6)=ROC(4,6)+1;
            end
            if predict_label(j)==5
                ROC(5,6)=ROC(5,6)+1;
            end
            if predict_label(j)==6
                ROC(6,6)=ROC(6,6)+1;
            end
            if predict_label(j)==7
                ROC(7,6)=ROC(7,6)+1;
            end
            if predict_label(j)==8
                ROC(8,6)=ROC(8,6)+1;
            end
         end
         
         if test_l(j)==7
            if predict_label(j)==1
                ROC(1,7)=ROC(1,7)+1;
            end
            if predict_label(j)==2
                ROC(2,7)=ROC(2,7)+1;
            end
            if predict_label(j)==3
                ROC(3,7)=ROC(3,7)+1;
            end
            if predict_label(j)==4
                ROC(4,7)=ROC(4,7)+1;
            end
            if predict_label(j)==5
                ROC(5,7)=ROC(5,7)+1;
            end
            if predict_label(j)==6
                ROC(6,7)=ROC(6,7)+1;
            end
            if predict_label(j)==7
                ROC(7,7)=ROC(7,7)+1;
            end
            if predict_label(j)==8
                ROC(8,7)=ROC(8,7)+1;
            end
         end
         
        if test_l(j)==8
            if predict_label(j)==1
                ROC(1,8)=ROC(1,8)+1;
            end
            if predict_label(j)==2
                ROC(2,8)=ROC(2,8)+1;
            end
            if predict_label(j)==3
                ROC(3,8)=ROC(3,8)+1;
            end
            if predict_label(j)==4
                ROC(4,8)=ROC(4,8)+1;
            end
            if predict_label(j)==5
                ROC(5,8)=ROC(5,8)+1;
            end
            if predict_label(j)==6
                ROC(6,8)=ROC(6,8)+1;
            end
            if predict_label(j)==7
                ROC(7,8)=ROC(7,8)+1;
            end
            if predict_label(j)==8
                ROC(8,8)=ROC(8,8)+1;
            end
         end
     end
     
     
     b1=ROC(1,1)+ROC(2,1)+ROC(3,1)+ROC(4,1)+ROC(5,1)+ROC(6,1)+ROC(7,1)+ROC(8,1);
     if b1==0
         b1=1;
         sos1=sos1+1;
     end
     for jj=1:8
       ROC1(jj,1)=ROC(jj,1)/b1;
     end
     
     b2=ROC(1,2)+ROC(2,2)+ROC(3,2)+ROC(4,2)+ROC(5,2)+ROC(6,2)+ROC(7,2)+ROC(8,2);
     if b2==0
         b2=1;
         sos2=sos2+1;
     end
     for jj=1:8
       ROC1(jj,2)=ROC(jj,2)/b2;
     end
     
     b3=ROC(1,3)+ROC(2,3)+ROC(3,3)+ROC(4,3)+ROC(5,3)+ROC(6,3)+ROC(7,3)+ROC(8,3);
     if b3==0
         b3=1;
         sos3=sos3+1;
     end
     for jj=1:8
       ROC1(jj,3)=ROC(jj,3)/b3;
     end
     
     b4=ROC(1,4)+ROC(2,4)+ROC(3,4)+ROC(4,4)+ROC(5,4)+ROC(6,4)+ROC(7,4)+ROC(8,4);
     if b4==0
         b4=1;
         sos4=sos4+1;
     end
     for jj=1:8
       ROC1(jj,4)=ROC(jj,4)/b4;
     end
     
     b5=ROC(1,5)+ROC(2,5)+ROC(3,5)+ROC(4,5)+ROC(5,5)+ROC(6,5)+ROC(7,5)+ROC(8,5);
     if b5==0
         b5=1;
         sos5=sos5+1;
     end
     for jj=1:8
       ROC1(jj,5)=ROC(jj,5)/b5;
     end
     
     b6=ROC(1,6)+ROC(2,6)+ROC(3,6)+ROC(4,6)+ROC(5,6)+ROC(6,6)+ROC(7,6)+ROC(8,6);
     if b6==0
         b6=1;
         sos6=sos6+1;
     end
     for jj=1:8
       ROC1(jj,6)=ROC(jj,6)/b6;
     end
     
     b7=ROC(1,7)+ROC(2,7)+ROC(3,7)+ROC(4,7)+ROC(5,7)+ROC(6,7)+ROC(7,7)+ROC(8,7);
     if b7==0
         b7=1;
         sos7=sos7+1;
     end
     for jj=1:8
       ROC1(jj,7)=ROC(jj,7)/b7;
     end
          
     b8=ROC(1,8)+ROC(2,8)+ROC(3,8)+ROC(4,8)+ROC(5,8)+ROC(6,8)+ROC(7,8)+ROC(8,8);
     if b8==0
         b8=1;
         sos8=sos8+1;
     end
     for jj=1:8
       ROC1(jj,8)=ROC(jj,8)/b8;
     end
  ROC2 = ROC2+ ROC1; 
end

% Mean accuracy
fprintf('Iteration\n');
meanACC=mean(ACC);

if (10-sos1)~=0
    meanROC(:,1)=ROC2(:,1)/(10-sos1);
else
    meanROC(:,1)=ROC2(:,1);
end

if (10-sos2)~=0
    meanROC(:,2)=ROC2(:,2)/(10-sos2);
else
    meanROC(:,2)=ROC2(:,2);
end

if (10-sos3)~=0
    meanROC(:,3)=ROC2(:,3)/(10-sos3);
else
    meanROC(:,3)=ROC2(:,3);
end

if (10-sos4)~=0
    meanROC(:,4)=ROC2(:,4)/(10-sos4);
else
    meanROC(:,4)=ROC2(:,4);
end

if (10-sos5)~=0
    meanROC(:,5)=ROC2(:,5)/(10-sos5);
else
    meanROC(:,5)=ROC2(:,5);
end

if (10-sos6)~=0
    meanROC(:,6)=ROC2(:,6)/(10-sos6);
else
    meanROC(:,6)=ROC2(:,6);
end

if (10-sos7)~=0
    meanROC(:,7)=ROC2(:,7)/(10-sos7);
else
    meanROC(:,7)=ROC2(:,7);
end

if (10-sos8)~=0
    meanROC(:,8)=ROC2(:,8)/(10-sos8);
else
    meanROC(:,8)=ROC2(:,8);
end

end