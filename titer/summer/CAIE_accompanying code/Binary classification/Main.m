% 3/14/2014 - Talayeh Razzaghi and Petros Xanthopoulos- Industral Engineering and Management Systems, University of central Florida - talayeh.razzaghi@gmail.com            
% Main Script    
%
% These parameters should be given by the user:                                                                                                                                   
% r: imbalanced ratio (r=0,...,45)                          
% w: window length    (w=10:5:100)  
% abtype: Abnormal type (e.g. Uptrend=1, Downtrend=2, Uphift=3, Downshift=4, Cyclic=5, Systematic=6, Stratification=7)
% t: parameter of abnormal pattern   for all patterns except stratification, let t=0.005:0.025:1.805;  and for stratification let t=0.005:0.025:0.4;                         
% Please indicate whether you would like to run SVM or WSVM by the variable mod,
% if mod==1 SVM is running, if mod==2, WSVM is running 
% Please indicate the type of abnormal pattern in GenerateData function (e.g. Uptrend, Downtrend, Upshift, Downshift, Cyclic, Sytematic, Stratification)                                      

clear all
close all
initial_folder=pwd;
%-----These parameters are set by the user----------------
mod = 2;
r = 45;               % highly imbalanced data
w = 10;
abtype = 1;          
t = 0.105;
%--------------------------------------------------------
A = [];  AB = [];  AC = [];  AD = [];
a =(50+1*r)*10;      % the size of Normal class based on the imbalanced ratio (r)
b =(50-1*r)*10;      % the size of abnormal class

% Generate Data
Data=GenerateData(w,t,abtype,a,b,mod);

% Repeating the experiment 10 times to get a non-variant and stable results
for i=1:10      
    [Sen, Spe, Acc, GMEAN] = wsvmmodel(Data,a,b,w,t,mod);
    A=[A; GMEAN];
    AB=[AB; Sen];
    AC=[AC; Spe];
    AD=[AD; Acc];
end

 Sensitivity=mean(AB);
 Specificity=mean(AC);
 Accuracy=mean(AD);
 Gmean=mean(A);
 
% % imagesc(t1,w1,A1);
% % Colormap gray;
% % set(gca,'YDir','normal');
% %   str=['0.005s';'0.105s';'0.205s';'0.305s','0.405s'];
% %    set(gca,'xticklabel',str,'fontname','symbol');
% %    %set(gca,'yticklabel',);
% %    months = ['10';'15';'20';'25';'30';'35';'40';'45';'50';'55';'60';'65';'70';'75';'80';'85';'90';'95';'100'];
% %    set(gca,'YTickLabel',months);
% %    
% %   xlabel('Systematic parameter');
% %   ylabel('Window length');