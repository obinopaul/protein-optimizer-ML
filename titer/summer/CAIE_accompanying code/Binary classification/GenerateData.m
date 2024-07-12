function Data = GenerateData(w,t,abtype,a,b,mod)
% 3/14/2014 - Talayeh Razzaghi and Petros Xanthopoulos - Industral Engineering and Management Systems, University of central Florida - talayeh.razzaghi@gmail.com            
%                                                                                                  
% INPUT:                                                                                                                                     
% w: window length     
% abtype: Abnormal type (e.g. Uptrend, Downtrend, Uphift,Downshift, Cyclic, Systematic, Stratification)
% t: parameter of abnormal pattern   
% a: the size of Normal class
% b: the size of abnormal class  
% mod: if mod==1 SVM is running elseif mod==2 WSVM is running                                     
                                     
% OUTPUT:                                                                                          
% Data: Time series data with label and weights, If we run SVM, the weights for all data samples are one.  

Data1 =[];
Data2 =[];
Data =[];
y = zeros(1,w);
s = 1:w;
 
 % Generates Normal data points
 for i = 1:a
     x = randn(1,w);
     Data1 = [Data1; x];
 end
   
 % Generates Abnormal data points
 for i = 1:b
     if abtype == 1
         y = randn(1,w) + t.*s;                       % Up trend(+)
     end
     if abtype == 2
        y = randn(1,w) - t.*s;                        % Downtrend(-)
     end
     if abtype == 3
       y=randn(1,w) + t*ones(1,w);                    % Up shift(+)
     end
     if abtype == 4
       y=randn(1,w) - t*ones(1,w);                    % Downshift(-)
     end
     if abtype == 5
       y=randn(1,w) + t*(-1).^s;                      % Systematic pattern
     end
      if abtype == 6
       y=randn(1,w) + t*cos(2*pi.*s/8);               % Cyclic pattern
     end
     if abtype == 7
       y=t*randn(1,w);                                % Stratification
     end               
     Data2 = [Data2; y];
 end
              
 Data=[Data1; Data2];                                 % Data is created
  
% label the data, gives 1 to Normal data and 0 to abnormal data
  Data(:,w+1) = [ones(1,a) zeros(1,b)];          

% Weights: when mod==2 (WSVM), assign all data points the weights as inverse of the class size otherwise assign all data points the weight equal to one (SVM)
if mod==2
    Data(:,w+2) = [1/a*ones(1,a) 1/b*ones(1,b)];
else
    siz = size(Data,1);
    Data(:,w+2)= ones(siz,1);               % Assign one to all examples when mod==1 (SVM)
end

end
 
 

