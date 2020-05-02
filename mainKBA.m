% 
% (c) 2019 Naoki Masuyama
% 
% These are the codes of Kernel Bayesian Adaptive Resonance Theory (KBA)
% proposed in "N. Masuyama, C. L. Loo, and F. Dawood, Kernel Bayesian 
% ART and ARTMAP, Neural Networks, vol. 98, pp. 76-86, November 2017."
% 
% Please contact "masuyama@cs.osakafu-u.ac.jp" if you have any problem.
% 

% clc
% clear
whitebg('black')


nData = 5000;

% 2D dataset
% data = corners(nData);
% data = crescentfullmoon(nData);
% data = halfkernel(nData);
% data = outlier(nData);
% data = twospirals(nData);
data = rings(nData);

DATA = [data(:,1) data(:,2)];


% Randamization
ran = randperm(size(DATA,1));
DATA = DATA(ran,:);


% scaling [0,1]
DATA = normalize(DATA,'range');


% Parameters of KBA
KBAnet.numClusters = 0;         % Number of clusters
KBAnet.weight      = [];        % Mean of cluster
KBAnet.ClusterAttribution = []; % Cluster attribution for each input
KBAnet.CountCluster = [];       % Counter for each cluster
KBAnet.numEpochs   = 1;         % Number of Epochs
KBAnet.maxNumClusters = 10000;  % Maximum number of clusters

KBAnet.maxCIM = 0.20;
KBAnet.kbrSig = 0.20;
KBAnet.cimSig = 0.30;




% Kernel Bayes ART ----------------------------
KBAnet = KBA(KBAnet, DATA);

figure(1);
myPlotKBART(DATA, KBAnet, 'KBA');
% ---------------------------------------------






