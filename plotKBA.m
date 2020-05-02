% 
% (c) 2019 Naoki Masuyama
% 
% These are the codes of Kernel Bayesian Adaptive Resonance Theory (KBA)
% proposed in "N. Masuyama, C. L. Loo, and F. Dawood, Kernel Bayesian 
% ART and ARTMAP, Neural Networks, vol. 98, pp. 76-86, November 2017."
% 
% Please contact "masuyama@cs.osakafu-u.ac.jp" if you have any problem.
% 
function plotKBA(DATA, net)

w = net.weight;
[N,D] = size(w);

colorNode = [
    [1 0 0]; 
    [0 1 0]; 
    [0 0 1]; 
    [1 0 1]; 
    [1 1 0];
    [0.8500 0.3250 0.0980];
    [0.9290 0.6940 0.1250];
    [0.4940 0.1840 0.5560];
    [0.4660 0.6740 0.1880];
    [0.3010 0.7450 0.9330];
    [0.6350 0.0780 0.1840];
];


plot(DATA(:,1),DATA(:,2),'cy.');
hold on;

for k = 1:N
    if D==2
        plot(w(k,1),w(k,2),'.','Color',colorNode(1,:),'MarkerSize',35);
    elseif D==3
        plot3(w(k,1),w(k,2),w(k,3),'.',colorNode(1,:),'MarkerSize',35);
    end
end

axis([0 1 0 1]);
hold off;
axis equal;
grid on;
    
end