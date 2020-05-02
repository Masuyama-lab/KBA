% 
% (c) 2019 Naoki Masuyama
% 
% These are the codes of Kernel Bayesian Adaptive Resonance Theory (KBA)
% proposed in "N. Masuyama, C. L. Loo, and F. Dawood, Kernel Bayesian 
% ART and ARTMAP, Neural Networks, vol. 98, pp. 76-86, November 2017."
% 
% Please contact "masuyama@cs.osakafu-u.ac.jp" if you have any problem.
% 
function net = KBA(net, patterns)


numClusters = net.numClusters;               % Number of clusters
weight = net.weight;                         % Mean of cluster
ClusterAttribution = net.ClusterAttribution; % Cluster attribution for each input
CountCluster = net.CountCluster;             % Counter for each cluster
numEpochs = net.numEpochs;                   % Number of Epochs
maxNumClusters = net.maxNumClusters;         % Maximum number of clusters
maxCIM = net.maxCIM;                         % Vigilance Parameter by CIM
kbrSig = net.kbrSig;                         % Kernel Bandwidth for Kernel Bayes Rule
cimSig = net.cimSig;                         % Kernel Bandwidth for CIM


% Classify and learn on each sample.
numSamples = size(patterns,1);

for epochNumber = 1:numEpochs
    
    for sampleNum = 1:numSamples
        
        % Current data sample
        pattern = patterns(sampleNum,:);
        
        
        % Kernel Bayes
        if sampleNum == 1 && epochNumber == 1
            KernelPo = nan(1, numClusters);
        else
            % Parameters for Kernel Bayes Rule
            paramKBR.Sig     = kbrSig;        % Kernel bandwidth  % Need to adjast Kernel bandwidth
            paramKBR.numNode = numClusters;      % Number of Clusters
            paramKBR.Eps     = 0.01/numClusters; % Scaling Factor
            paramKBR.Delta   = 2*numClusters;    % Scaling Factor
            paramKBR.gamma   = ones(size(pattern, 1),1) / size(pattern, 1);  % Scaling Factor
            paramKBR.prior   = CountCluster' / sum(CountCluster);    % Prior Probability
            
            % Kernel Bayes Rule
            [KernelPo, ~] = KernelBayesRule(pattern, weight, paramKBR); % return [PostPr, posteriorMean]
        end
        
        [~, sortedClusters] = sort(-KernelPo);
        
        resonance = false;
        numSortedClusters = length(sortedClusters);
        currentSortedIndex = 1;
        
        % only for first cluster
        if numSortedClusters == 0 && epochNumber == 1
           % Add Cluster
           numClusters                      = numClusters + 1;
           weight(numClusters,:)            = pattern;
           ClusterAttribution(1, sampleNum) = 1;
           CountCluster(1, sampleNum)       = 1;

           resonance = true;
        end
        
        
        
        while ~resonance
           
           bestCluster = sortedClusters(currentSortedIndex);
           bestWeight = (CountCluster(1, bestCluster) * weight(bestCluster,:) + pattern) / (CountCluster(1, bestCluster) + 1);
           
           % Calculate CIM between winner cluster and pattern for Vigilance Test
           bestCIM = CIM(pattern, bestWeight, cimSig);
           
           % Vigilance Test
           if bestCIM <= maxCIM
               % Match Success       
               % Update Parameters
               weight(bestCluster,:)            = bestWeight;
               ClusterAttribution(1, sampleNum) = bestCluster;
               CountCluster(1, bestCluster)     = CountCluster(1, bestCluster) + 1;
               
               resonance = true;
           else
               % Match Fail
               if(currentSortedIndex == numSortedClusters)  % Reached to maximum number of generated clusters
                    if(currentSortedIndex == maxNumClusters)    % Reached to defined muximum number of clusters
                        ClusterAttribution(1, sampleNum) = -1;
                        fprintf('WARNING: The maximum number of categories has been reached.\n');
                        resonance = true;
                    else
                        % Add Cluster
                        numClusters                      = numClusters + 1;
                        weight(numClusters,:)            = pattern;
                        ClusterAttribution(1, sampleNum) = numClusters;
                        CountCluster(1, numClusters)     = 0;
                        CountCluster(1, numClusters)     = CountCluster(1, numClusters) + 1;

                        resonance = true;
                    end
               else
                   currentSortedIndex = currentSortedIndex + 1;    % Search another cluster orderd by sortedClusters
               end
               
           end % end Vigilance Test
           
        end % end resonance
        
    end % end numSample
end % end Epochs


net.numClusters = numClusters;               % Number of clusters
net.weight = weight;                         % Mean of cluster
net.ClusterAttribution = ClusterAttribution; % Cluster attribution for each input
net.CountCluster = CountCluster;             % Counter for each cluster
net.maxNumClusters = maxNumClusters;         % Maximum number of clusters

end



% Kernel Bayes Rule
function [Po, posteriorMean] = KernelBayesRule(pattern, weight, paramKBR)

% Xi : pattern
% Yj : weight
% Pr : prior probability of Y


meanU = mean(pattern,1);

% Parameters for Kernel Bayes Rule
Sig = paramKBR.Sig;           % Kernel bandwidth
numNode = paramKBR.numNode;   % Number of Nodes
Eps = paramKBR.Eps;           % Scaling Factor
Delta = paramKBR.Delta;       % Scaling Factor
gamma = paramKBR.gamma;       % Scaling Factor
Pr = paramKBR.prior;          % Prior Probability


% Calculate Gram Matrix
Gy = Gramian(weight, weight, Sig); % Gy
Gx = Gramian(Pr, Pr, Sig);         % Gx


m_hat = zeros(numNode,1);
tmp = zeros(size(Pr,1),size(pattern,1));
for i=1:size(Pr,1)
   for j=1:size(pattern,1)
       tmp(i,j) = gamma(j) * gaussian_kernel(pattern(j,:), Pr(i,:), Sig); % kx(.,Pr)
   end
   m_hat(i) = sum(tmp(i,:),2);
end

mu_hat = numNode \ (Gx + numNode * Eps * eye(numNode)) * m_hat;
Lambda = diag(mu_hat);
LG = Lambda * Gy; % \Lambda * Gy
R = LG \ (LG^2 + Delta * eye(numNode)) * Lambda;
ky = gaussian_kernel(weight, meanU, Sig);
tmp_m_hatQ = R * ky;
tmp_m_hatQ( tmp_m_hatQ < 0 ) = 0;

Po = tmp_m_hatQ / sum(tmp_m_hatQ); % Posterior Probability  m_hatQ
posteriorMean = weight'*Po; % Estimated Mean

end


% Gram Matrix
function gram = Gramian(X1, X2, sig)
a=X1'; b=X2';
if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))];  
  b = [b; zeros(1,size(b,2))];  
end 
aa=sum(a.*a); bb=sum(b.*b); ab=a'*b;  
D = sqrt(repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab);

gram = exp(-(D.^2 / (2 * sig.^2)));
end


% Correntropy induced Metric (Gaussian Kernel based)
function cim = CIM(X,Y,sig)
% X : 1 x n
% Y : m x n
[n, att] = size(Y);
g_Kernel = zeros(n, att);

for i = 1:att
    g_Kernel(:,i) = GaussKernel(X(i)-Y(:,i), sig);
end

ret0 = GaussKernel(0, sig);
ret1 = mean(g_Kernel, 2);

cim = sqrt(ret0 - ret1)';
end

% Gaussian Kernel
function g_kernel = GaussKernel(sub, sig)
g_kernel = exp(-sub.^2/(2*sig^2));
% g_kernel = 1/(sqrt(2*pi)*sig) * exp(-sub.^2/(2*sig^2));
end


% Gaussian Kernel
function g_kernel = gaussian_kernel(X, W, sig)
nrm = sum(bsxfun(@minus, X, W).^2, 2);
g_kernel = exp(-nrm/(2*sig^2));
end



