function data = rings(nData)

ringD(1) = floor(nData*0.111);
ringD(2) = floor(nData*0.333);
ringD(3) = nData-ringD(1)-ringD(2);

% 3 rings
rmin(1) = 0; rmax(1) = 2;
rmin(2) = 4; rmax(2) = 6;
rmin(3) = 8; rmax(3) = 10;


% 2 rings
% rmin(1) = 0; rmax(1) = 0;
% rmin(2) = 5; rmax(2) = 10;
% rmin(3) = 15; rmax(3) = 20;



% small ring
rMIN = rmin(1);
rMAX = rmax(1);
thetalist = rand(ringD(1),1)*2*pi;
rlist = rand(ringD(1),1)*(rMAX-rMIN)+rMIN;
x = zeros(ringD(1),2);
for i = 1:ringD(1)
    x(i,1) = rlist(i)*cos(thetalist(i));
    x(i,2) = rlist(i)*sin(thetalist(i));
end
tmpD1 = x;
tmpL1 = zeros(size(x,1),1);

% mid ring
rMIN = rmin(2);
rMAX = rmax(2);
thetalist = rand(ringD(2),1)*2*pi;
rlist = rand(ringD(2),1)*(rMAX-rMIN)+rMIN;
x = zeros(ringD(2),2);
for i = 1:ringD(2)
    x(i,1) = rlist(i)*cos(thetalist(i));
    x(i,2) = rlist(i)*sin(thetalist(i));
end
tmpD2 = x;
tmpL2 = zeros(size(x,1),1)+1;

% large ring
rMIN = rmin(3);
rMAX = rmax(3);
thetalist = rand(ringD(3),1)*2*pi;
rlist = rand(ringD(3),1)*(rMAX-rMIN)+rMIN;
x = zeros(ringD(3),2);
for i = 1:ringD(3)
    x(i,1) = rlist(i)*cos(thetalist(i));
    x(i,2) = rlist(i)*sin(thetalist(i));
end
tmpD3 = x;
tmpL3 = zeros(size(x,1),1)+2;


IMAGES = [tmpD1; tmpD2; tmpD3];
LABELS = [tmpL1; tmpL2; tmpL3];

% randamize
ran = randperm(size(LABELS,1));
IMAGES = IMAGES(ran,:);
LABELS = LABELS(ran,:);

data = [IMAGES, LABELS];

end






