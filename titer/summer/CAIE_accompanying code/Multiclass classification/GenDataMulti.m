function Data = GenDataMulti(w,n,m,t1,t2,t3,t4,t5,t6,t7,mod)
% 3/14/2014 - Talayeh Razzaghi and Petros Xanthopoulos- Industral Engineering and Management Systems, University of central Florida - talayeh.razzaghi@gmail.com            
%                                                                                                  
% INPUT:   
% w: window length    
% t1-t7: parameters of abnormal pattern    
% m : the size of Normal class
% n : the size of abnormal class for each abnormal pattern (We assume that all abnormal class size are equal                        
% mod: if mod==1 SVM is running elseif mod==2 WSVM is running                                     
                                     
% OUTPUT:                                                                                          
% Data: Time series data with label and weights, If we run SVM, the weights for all data samples are one.  

 Data1 = [];
 Data2 = [];
 Data = [];

 s = 1:w;
 
% Generates Normal data points
 for i=1:m
  x=randn(1,w);                  % Normally distributed pseudorandom numbers with w features (culomns) and size m (rows)
  Data1=[Data1; x];          
 end

% Generate Downtrend pattern 
for i=1:n
  y = randn(1,w)-t1.*s;            %For example: y=randn(1,w)-0.05.*t; 
  Data2 = [Data2; y]; 
end
 
% Uptrend pattern 
 for i=1:n
  y = randn(1,w)+t2.*s;            %For example: y=randn(1,w)+0.05.*s; 
  Data2 = [Data2; y]; 
 end
 
% Systematic pattern  
 for i=1:n
  y = randn(1,w)+t3*(-1).^s;        %For example: y=randn(1,w)+0.5*(-1).^s; 
  Data2 = [Data2; y]; 
 end
  
% Downshift
 for i=1:n
  y = randn(1,w)-t4*ones(1,w);      %For example: y=randn(1,w)-0.5*ones(1,w);                   
  Data2 = [Data2; y]; 
 end
 
% Upshift
 for i=1:n
  y = randn(1,w)+t5*ones(1,w);      %For example: y=randn(1,w)+0.5*ones(1,w); 
  Data2=[Data2; y]; 
 end
 
% Cyclic Pattern
 for i=1:n
     y = randn(1,w)+t6*cos(2*pi.*s/8);          %For example: y=randn(1,w)+0.05*cos(2*pi.*s/8);
     Data2 = [Data2; y];
 end
 
% Stratification Pattern 
 for i=1:n
     y = t7*randn(1,w);
     Data2 = [Data2; y];
 end
 
 Data = [Data1; Data2];
 
 v=1*n;   % The weights are adjustable
 
 % label the data, it gives 1 to Normal data, 2 to Downtrend data, 3 to Uptrend, 4 to Systematic, 5 to Downshift, 6 to Downshift, 7 to Cyclic, and 8 to Stratification
 Data(:,w+1) = [ones(1,m) 2*ones(1,n) 3*ones(1,n) 4*ones(1,n) 5*ones(1,n) 6*ones(1,n) 7*ones(1,n)  8*ones(1,n)];  % class label
 
 % Weights: when mod==2 (WSVM), assign all data points the weights as inverse of the class size otherwise assign all data points the weight equal to one (SVM)
 if mod==2
     Data(:,w+2) = [1/m*ones(1,m) 1/v*ones(1,n) 1/v*ones(1,n) 1/v*ones(1,n) 1/v*ones(1,n) 1/v*ones(1,n) 1/v*ones(1,n)  1/v*ones(1,n)];  %WSVM
 else
     Data(:,w+2) = [ones(1,m) ones(1,n) ones(1,n) ones(1,n) ones(1,n) ones(1,n) ones(1,n)  ones(1,n)];            %SVM
 end
 
end
 
 

