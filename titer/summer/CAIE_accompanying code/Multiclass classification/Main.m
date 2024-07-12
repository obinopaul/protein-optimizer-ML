% 3/14/2014 - Talayeh Razzaghi and Petros Xanthopoulos- Industral Engineering and Management Systems, University of central Florida - talayeh.razzaghi@gmail.com            
% Main Script    
%
% These parameters should be given by the user:                                                                                                                                                        
% w: window length    (w=10:5:100)                              
% t: parameter of abnormal pattern   for all patterns except stratification, let t=0.005:0.025:1.805;  and for stratification let t=0.005:0.025:0.4;   
% m: the number of Normal examples  
% n: the number of Abnormal examples in each abnormal pattern (We assume that all abnormal class size are equal)
% The parameters m and n are indicating of the degree of imbalancedness degree
% Please indicate whether you would like to run SVM or WSVM by the variable mod,
% if mod==1 SVM is running, if mod==2, WSVM is running 
% file mt: shows the matrix 1*7 of abnormal parameter for seven abnrmal patterns (Downtrend, Uptrend, Sytematic, Downshift, Upshift, Cyclic, Stratification)                                      

clear all
close all
initial_folder=pwd;
%-----These parameters are set by the user----------------
m = 951;    %  m=909;    n=13;
n = 7;
w = 25;
mod = 2;
load multiclass;
%--------------------------------------------------------
t1=mt(1,1);        % Downtrend pattern
t2=mt(1,2);        % Uptrend pattern
t3=mt(1,3);        % Systematic pattern
t4=mt(1,4);        % Down shift
t5=mt(1,5);        % Up shift
t6=mt(1,6);        % Cyclic Pattern
t7=mt(1,7);        % Stratification Pattern
A = 0;
AB = zeros(8,8);

% Generate data
data = Genweight_datamulti(w,n,m,t1,t2,t3,t4,t5,t6,t7,mod);

% Repeating the experiment 10 times to get a non-variant and stable results
for i=1:10
    [meanACC, meanROC] = wsvmmodelmulti(data,w,n,m,mod);
    A = A + meanACC;
    AB = AB + meanROC;
end

 ROC = AB/10;
 Accuracy = A/10;





    
 



