%%
clc
clear all
close all
clf
%MyPanorama()

%%
%function [pano] = MyPanorama()
    Nbest = 300;
    myFolder = '/Users/matthewwong/UMD/Fall-20/CMSC426/Project 2/P2 2/Images/Set3';
    filePattern = fullfile(myFolder, '*.jpg');
    imgFiles = dir(filePattern);
    numFiles = length(imgFiles);
    % Must load images from ../Images/Input/
    img1 = imread(char('/Users/matthewwong/UMD/Fall-20/CMSC426/Project 2/P2 2/Images/Set3/' + string(imgFiles(1).name)));
    gray1 = rgb2gray(img1);
    [x1,y1] = ANMS(gray1, Nbest);
    features1 = fdesc(x1,y1,gray1);
    % change based on image
    tols = [0, .4, .35, .5, .5, .5, .5, .5]
    
        
    for file = 2:(numFiles)
        disp('Processing Image' + string(file))
        img2 = imread(char('/Users/matthewwong/UMD/Fall-20/CMSC426/Project 2/P2 2/Images/Set3/' + string(imgFiles(file).name)));
        gray2 = rgb2gray(img2);
        
        % ANMS
        [x2, y2] = ANMS(gray2,Nbest);
        
        % Feature Descriptors
        features2 = fdesc(x2,y2,gray2);
        
        % Feature Matching
        [matchedPoints1,matchedPoints2] = fmatch(features1,features2,x1,x2,y1,y2, tols(file));
        
        figure(4);
        showMatchedFeatures(gray1, gray2, matchedPoints1, matchedPoints2, 'montage')
        
        % RANSAC
        [h] = ransac(length(matchedPoints1),matchedPoints1,matchedPoints2, 50, img1, img2);
        
        h = h';
        normalize = h(3,3);
        h = h ./ normalize;
 
        composite_blend = 1.4;

        transform = maketform('projective', h);
        [IleftT, xdataimT, ydataimT]=imtransform(img2, transform, 'XYScale',1);
        xdataout=[min(1,xdataimT(1)) max(size(img2,2),xdataimT(2))];
        ydataout=[min(1,ydataimT(1)) max(size(img1,1),ydataimT(2))];
        comp1=imtransform(img2,transform,'nearest','XData',xdataout,'YData',ydataout,'XYScale',1);
        comp2=imtransform(img1,maketform('affine',eye(3)),'nearest','XData',xdataout,'YData',ydataout);

        imshow(comp1)
        imshow(comp2)
        composite = comp1;
        [row, col, rgb] = size(composite);
        turn = 0;
        for i = 1:row
            for j = 1:col
                for k = 1:rgb
                    if (composite(i, j, k) == 0)
                        composite(i, j, k) = comp2(i, j, k);  
                    end
                end
            end
        end
        %figure()
        imshow(composite)
        
        
        
        
        
        % set for next iteration
        img1 = composite;
        gray1 = rgb2gray(composite);
        [x1, y1] = ANMS(gray1,Nbest);
        features1 = fdesc(x1,y1,gray1);

    end
    


%% ANMS
function [x,y, Nbest] = ANMS(img, Nbest)
    Cimg = cornermetric(img);
    cornerpeaks = imregionalmax(Cimg);
    corner_idx = find(cornerpeaks == true);

    [y, x] = ind2sub(size(Cimg), corner_idx);


    Nstrong = size(x,1); %Number of strong corner points
    radius = Inf(Nstrong, 1); %Initialize radius = inf

    for i=1:Nstrong
        for j=1:Nstrong
           if (Cimg(y(j), x(j)) > Cimg(y(i), x(i))) 
               ED = (y(j)-y(i))^2 + (x(j)-x(i))^2;

               if (ED < radius(i))
                   radius(i) = ED;
               end
           end
        end 
    end

    [radius_value, radius_idx] = sort(radius, 'descend');
    if (length(x) < Nbest)
        Nbest = length(x)
    end

    x = x(radius_idx(1:Nbest));
    y = y(radius_idx(1:Nbest));

    figure();
    imshow(img);
    hold on;
    plot(x, y, 'r.');
    hold off;
    [x, y];
end

%% Feature descriptor
function [features] = fdesc(x, y, img)
    features = [];
    for i=1:length(x)
        %taking 40x40 centered region
        row1 = y(i)-20;
        col1 = x(i)-20;
        zone = imcrop(img, [col1 row1 40 40]);
        filter = fspecial('gaussian');
        %applying gaussian filter
        zone = imfilter(zone, filter, 'replicate');
        %subsample
        zone = imresize(zone, [8 8]);
        %flatten to vector
        zone = reshape(zone, [64 1]);

        %standardize feature descriptor
        zone = zone - mean(zone(:));
        zone = cast(zone,'double');
        zone = zone / std(zone(:));

        features = [features zone];
    end
end

%% Sum of Square Difference
function ssdOutput = ssd(a,b)
    ssdOutput = sum((a(:)-b(:)).^2);
end

%% Feature Matching
function [matchedPoints1, matchedPoints2] = fmatch(ft1,ft2,xbest1,xbest2,ybest1,ybest2, tol)
    matchedPoints1 = zeros(1,2);
    matchedPoints2 = zeros(1,2);
    matchedPoints1(:) = [];
    matchedPoints2(:) = [];
    for i = 1:length(xbest1)
        for j = 1:length(xbest1)
            ssdOut(j) = ssd(ft1(:,i),ft2(:,j));
        end
        [low,I] = sort(ssdOut,'ascend');
        lowestMatch = low(1);
        secondMatch = low(2);
        if((lowestMatch/secondMatch) < tol)
            matchedPoints1=vertcat(matchedPoints1,[xbest1(i) ybest1(i)]);
            matchedPoints2=vertcat(matchedPoints2,[xbest2(I(1)) ybest2(I(1))]);
        end
    end
end

%% RANSAC

function [h] = ransac(num_matched,matched1,matched2,max_iters, img, img2)
    N = 1000;
    s = 4;
    sample_ct = 0;
    p = 0.99;
    maxNum = -1;
    
    while(N > sample_ct || sample_ct < max_iters)
        perm = randperm(num_matched,4);
		indice_1 = perm(1);
		indice_2 = perm(2);
		indice_3 = perm(3);
		indice_4 = perm(4);

		xSource = [matched1(indice_1,1);matched1(indice_2,1);matched1(indice_3,1);matched1(indice_4,1)];
		ySource = [matched1(indice_1,2);matched1(indice_2,2);matched1(indice_3,2);matched1(indice_4,2)];
        
        xDest = [matched2(indice_1,1);matched2(indice_2,1);matched2(indice_3,1);matched2(indice_4,1)];
        yDest = [matched2(indice_1,2);matched2(indice_2,2);matched2(indice_3,2);matched2(indice_4,2)];
        
        h = est_homography(xDest,yDest,xSource,ySource);
        
        % Get Number of Inliers
        [numInlier, xSrc, ySrc, xDest, yDest] = getInliers(num_matched,matched1,matched2,h,50);
        
        if(numInlier >= maxNum)
            maxNum = numInlier;
            maxXSrc = xSrc;
            maxYSrc = ySrc;
            maxXDest = xDest;
            maxYDest = yDest;
        end
        
        e = 1-numInlier/num_matched;
        N = log(1-p)/log(1-power(1-e,s));
        sample_ct = sample_ct + 1;
    end
    if(numInlier/num_matched < 0.5)
        error("inlier ratio < 50%")
    end
    
    %Showing Matched Features after RANSAC (Removed Outliers)
    inlierPoints = zeros(1,2);
    inlierPoints(:) = [];
    inlierPoints2 = zeros(1,2);
    inlierPoints2(:) = [];
    for i = 1:numInlier
        inlierPoints = vertcat(inlierPoints,[maxXSrc(i) maxYSrc(i)]);
        inlierPoints2 = vertcat(inlierPoints2,[maxXDest(i) maxYDest(i)]);
    end
    figure(5);
    showMatchedFeatures(img, img2, inlierPoints, inlierPoints2, 'montage');
    
    h = est_homography(maxXSrc,maxYSrc,maxXDest,maxYDest);
end

%% Get Inliers

function [numInliers, xSrc, ySrc, xDest, yDest] = getInliers(num, pts1, pts2, h, tol)
    numInliers = 0;
    for i = 1:num
        % point on img1
        x = pts1(i,1);
        y = pts1(i,2);
        % point on img2
        x2 = pts2(i,1);
        y2 = pts2(i,2);
        
        % Apply H to points on img1 to find "points" on img2
        [xRes, yRes] = apply_homography(h,x,y);
        
        % Find the difference b/w applied points and actual matched point 
        dist = ssd([xRes,yRes],[x2,y2]);
        
        % if distance b/w 
        if(dist < tol)
            numInliers = numInliers + 1;
            xSrc(i) = x;
            ySrc(i) = y;
            xDest(i) = x2;
            yDest(i) = y2;
        end
    end
end