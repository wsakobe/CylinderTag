addpath('./Data')
clear

obj = VideoReader('result4.avi');%输入视频位置
numFrames = obj.NumFrames;% 帧的总数
r=5;
template=generateTemplate(r);

%% global announcement
global visited square graph ptList_refine

for frame = 1:numFrames
    %% refreshing
    judge=[];
    square=[];
    ptList=[];
    feature=[];
    
    %% read from video
    img = im2double(rgb2gray(read(obj,frame)));%读取当前帧
%     img=(imread('1.bmp'));

    %% Fast corner detection + NMS
%     corners=detectFASTFeatures(img,'MinQuality',0.01,'MinContrast',0.15);
%     ptList = corners.Location;
%     boundary = (ptList(:,1)<=r+2)+(ptList(:,2)<=r+2)+(ptList(:,1)>=size(img,2)-r-2)+(ptList(:,2)>=size(img,1)-r-2);
%     ptList(boundary~=0,:)=[];
% 
%     for i=1:size(ptList,1)
%         for j=1:8
%             score(i,j)=corr2(squeeze(template(j,:,:)),im2double(img(ptList(i,2)-r:ptList(i,2)+r,ptList(i,1)-r:ptList(i,1)+r)));
%         end
%         score(i,:)=sort(score(i,:),2,'descend');
%         if score(i,1)<0.7 || score(i,8)>0
%             judge(i)=1;
%         end
%     end
%     ptList=ptList(judge==0,:);
%     ptList=[ptList(:,2) ptList(:,1)];
    
    %% subpixel refine
    img_resize=imresize(img,0.5);
    img_binary = adaptiveThreshold(img_resize,5);
    [Gxr,Gyr] = imgradientxy(img_resize);
    Gpowr = (Gxr.^2+Gyr.^2).^0.5;
    Gangle = rad2deg(atan2(Gyr, Gxr));
    polygonArea = areaGrowth(img_binary);
    graph=[];
%     imshow(img_resize)
%     hold on

    for i=1:size(polygonArea,2)
        dis=[];
        pos=[];
        angle_col=[];
        angle_center=[];
        angle_diff=[];
        corner=[];
        queue=polygonArea{i};
        maxGpowr=max(max(Gpowr(queue(:,1),queue(:,2))));
        ilg=zeros(1,size(queue,1));
        for iter=1:size(queue,1)
            angle_center(iter)=rad2deg(atan2(queue(iter,1)-mean(queue(:,1)),queue(iter,2)-mean(queue(:,2))));
            angle_col(iter)=Gangle(queue(iter,1),queue(iter,2));
            if Gpowr(queue(iter,1),queue(iter,2))<0.5*maxGpowr
                ilg(iter)=1;
            end
        end
        boundary=queue(ilg==0,:);
        boundary_grad=angle_col(ilg==0);
        if size(boundary_grad,2)<4
            continue;
        end
        boundary_class=kmeans(boundary_grad',4,'Start','sample','Replicates',10);
%         scatter(boundary(boundary_class==1,2),boundary(boundary_class==1,1),20,'c','filled')
%         scatter(boundary(boundary_class==2,2),boundary(boundary_class==2,1),20,'r','filled')
%         scatter(boundary(boundary_class==3,2),boundary(boundary_class==3,1),20,'b','filled')
%         scatter(boundary(boundary_class==4,2),boundary(boundary_class==4,1),20,'g','filled')
        k(1,:)=polyfit(boundary(boundary_class==1,2),boundary(boundary_class==1,1),1);
        k(2,:)=polyfit(boundary(boundary_class==2,2),boundary(boundary_class==2,1),1);
        k(3,:)=polyfit(boundary(boundary_class==3,2),boundary(boundary_class==3,1),1);
        k(4,:)=polyfit(boundary(boundary_class==4,2),boundary(boundary_class==4,1),1);
%         x1=min(boundary(boundary_class==1,2)):0.1:max(boundary(boundary_class==1,2));
%         y1=k(1,1)*x1+k(1,2);
%         plot(x1,y1);
%         x2=min(boundary(boundary_class==2,2)):0.1:max(boundary(boundary_class==2,2));
%         y2=k(2,1)*x2+k(2,2);
%         plot(x2,y2);
%         x3=min(boundary(boundary_class==3,2)):0.1:max(boundary(boundary_class==3,2));
%         y3=k(3,1)*x3+k(3,2);
%         plot(x3,y3);
%         x4=min(boundary(boundary_class==4,2)):0.1:max(boundary(boundary_class==4,2));
%         y4=k(4,1)*x4+k(4,2);
%         plot(x4,y4);
        for iter=1:3
            for j=iter+1:4
                x_int=(k(j,2)-k(iter,2))/(k(iter,1)-k(j,1));
                y_int=k(iter,1)*x_int+k(iter,2);
                if norm(mean(queue)-[y_int x_int])>80 || x_int <= r || y_int <= r || x_int > size(img_binary,2)-r-2 || y_int > size(img_binary,1)-r-2
                    continue;
                end
                corner=[corner; y_int x_int];
            end
        end
        if size(corner,1)<4
            continue;
        end
        if size(corner,1)>4
            dis=vecnorm(corner-mean(queue),2,2);
            [~,pos]=sort(dis);
            corner=corner(pos(1:4),:);
        end
        for i=1:4
            angle(i)=rad2deg(atan2(corner(i,2)-mean(queue(:,2)),corner(i,1)-mean(queue(:,1))));
        end
        [~,pos]=sort(angle);
        corner=corner(pos(1:4),:);
        square_number=polyarea(corner(:,1),corner(:,2));
        if size(queue,1)<0.8*square_number || size(queue,1)>1.5*square_number
            continue;
        end
        ptList=[ptList; corner];
        graph(size(ptList,1)-3,size(ptList,1)-2)=1;
        graph(size(ptList,1)-2,size(ptList,1)-1)=1;
        graph(size(ptList,1)-1,size(ptList,1))=1;
        graph(size(ptList,1),size(ptList,1)-3)=1;
        graph(size(ptList,1)-2,size(ptList,1)-3)=1;
        graph(size(ptList,1)-1,size(ptList,1)-2)=1;
        graph(size(ptList,1),size(ptList,1)-1)=1;
        graph(size(ptList,1)-3,size(ptList,1))=1;
%         scatter(queue(:,2),queue(:,1),10,'b','filled')
%         
%         scatter(mean(queue(:,2)),mean(queue(:,1)),30,'r','filled')
%         text(queue(:,2),queue(:,1),num2str(angle_diff(:),3))
%         text(mean(queue(:,2)),mean(queue(:,1)),[num2str(i)]);
    end
    
    ptList=ptList*2;
    boundary = (ptList(:,1)<=r+2)+(ptList(:,2)<=r+2)+(ptList(:,1)>=size(img,1)-r-2)+(ptList(:,2)>=size(img,2)-r-2);
    ptList(boundary~=0,:)=[];
    graph(boundary~=0,:)=[];
    graph(:,boundary~=0)=[];
    
    [Gx,Gy] = imgradientxy(img);
    Gpow = (Gx.^2+Gy.^2).^0.5;
    ptList_refine = cornerSubPix(Gx, Gy, ptList, size(img,2), size(img,1));
    figure(1)
    imshow(img)
    hold on
    scatter(ptList_refine(:,2),ptList_refine(:,1),15,'b','filled')
    
    NMS_matrix=zeros(size(img,1),size(img,2));
    for i=1:size(ptList_refine,1)
        NMS_matrix(round(ptList_refine(i,1)),round(ptList_refine(i,2)))=i;
    end
    NMS_matrix(imdilate(NMS_matrix,strel('square',3))~=NMS_matrix)=0;
    ptList_refine=ptList_refine(NMS_matrix(NMS_matrix~=0),:);
    graph=graph(NMS_matrix(NMS_matrix~=0),NMS_matrix(NMS_matrix~=0));

    %% edge extraction
%     graph=zeros(size(ptList_refine,1));
%     cnt=zeros(size(ptList_refine,1),1);
%     for i = 1:size(ptList_refine,1)-1
%         for j = i+1:size(ptList_refine,1)
%             if cnt(i)==2
%                 break;
%             end
%             if cnt(j)==2
%                 continue;
%             end
%             if judgeEdge(ptList_refine(i,:), ptList_refine(j,:), Gpow)
%                 cnt(i)=cnt(i)+1;
%                 cnt(j)=cnt(j)+1;
%                 addEdge(i,j);
%             end
%         end
%     end
    
%     figure(1)
%     imshow(img)
%     hold on
%     for i=1:size(ptList_refine,1)-1
%         for j=i+1:size(ptList_refine,1)
%             if graph(i,j)==1
%                 X=[ptList_refine(i,2) ptList_refine(j,2)];
%                 Y=[ptList_refine(i,1) ptList_refine(j,1)];
%                 line(X,Y,'Color','y','linewidth',3);
%             end
%         end
%     end
%     scatter(ptList_refine(:,2),ptList_refine(:,1),15,'r','filled')
    
%     ptList_refine(cnt<2,:)=[];
%     graph(cnt<2,:)=[];
%     graph(:,cnt<2)=[];
    
    %% contour judgment
    visited=zeros(size(ptList_refine,1),1);
    for i = 1:size(ptList_refine,1)
        if ~visited(i)
            visited(i)=1;
            dfs(i,i,i);
        end
    end
    
%     figure(2)
%     imshow(img)
%     hold on
%     for j=1:size(square,1)
%         text(ptList_refine(square(j,1),2),ptList_refine(square(j,1),1),num2str(j),'fontsize',16,'Color','r');
%     end
    
    %% feature organization
    for i=1:size(square,1)-1
        for j=i+1:size(square,1)
            if judgeFeature(square(i,:), square(j,:))
                feature = [feature; organizeFeature(square(i,:), square(j,:))];
            end
        end
    end
    
%     figure(1)
%     imshow(img)
%     hold on
    for i=1:size(feature,1)
        X=[ptList_refine([feature(i,1:3) feature(i,8) feature(i,5:7) feature(i,4) feature(i,1)],2)];
        Y=[ptList_refine([feature(i,1:3) feature(i,8) feature(i,5:7) feature(i,4) feature(i,1)],1)];
        line(X,Y,'Color','y','linewidth',2);
%         for j=1:size(feature,2)
%             text(ptList_refine(feature(i,j),2),ptList_refine(feature(i,j),1),num2str(feature(i,j)));
%         end
    end
    scatter(ptList_refine(:,2),ptList_refine(:,1),20,'b','filled');
    
    %% feature recovery
    [ID, center, direc] = extractFeature(img,feature);
    
    for i=1:size(feature,1)
        text(ptList_refine(feature(i,4),2),ptList_refine(feature(i,4),1),num2str(ID(i)),'fontsize',16,'Color','y');
    end
    
    %% split each marker
    markers={};
    cnt=0;
    visit=zeros(1,size(feature,1));
    for i=1:size(feature,1)
        if visit(i)
            continue;
        end
        marker_now=i;
        visit(i)=1;
        dist=vecnorm(center(i,:)-center(:,:),2,2);
        [dist, pos]=sort(dist);
        for j=2:size(feature,1)
            if dist(j)<15 * norm(ptList_refine(feature(i,1),:)-ptList_refine(feature(i,2),:)) && abs(direc(i)-direc(pos(j))) < 2 && ~visit(pos(j))
                visit(pos(j))=1;
                marker_now=[marker_now pos(j)];
                continue;
            end
            break;
        end
        cnt=cnt+1;
        markers{cnt}=marker_now;
    end
    
    %% obtain marker IDs
    ID_lib=load('ID.mat');
    for i=1:size(markers,2)
        code=-1*ones(1,size(ID_lib.ID,2));
        if abs(direc(markers{1,i}(1))) < 135 && abs(direc(markers{1,i}(1))) > 45
            [~,pos]=sort(center(markers{1,i},2));
        else
            [~,pos]=sort(center(markers{1,i},1));
        end
        code(1)=ID(markers{1,i}(pos(1)));
        last_pos = 1;
        for j=2:size(pos,1)
            dist_feature=norm(center(markers{1,i}(pos(j)),:)-center(markers{1,i}(pos(j-1)),:));
            feature_length=norm(ptList_refine(feature(markers{1,i}(j),1),:)-ptList_refine(feature(markers{1,i}(j),2),:));
            code(last_pos + round(dist_feature/(2*feature_length)))=ID(markers{1,i}(pos(j)));
            last_pos = last_pos + round(dist_feature/(2*feature_length));
        end
        [ID_num, ID_pos, ID_direction]=findID(code,ID_lib.ID);
        
%         for it=1:size(markers{1,i},2)
%             if ID_num==-1
%                 text(ptList_refine(feature(markers{1,i}(it),1),2),ptList_refine(feature(markers{1,i}(pos(it)),1),1)-10,'Fail to recover ID','fontsize',12,'Color','r');
%             else
%                 text(ptList_refine(feature(markers{1,i}(it),1),2),ptList_refine(feature(markers{1,i}(pos(it)),1),1)-10,['ID:' num2str(ID_num) ' Pos:' num2str(mod(ID_pos + ID_direction * (it - 1) + size(ID_lib.ID, 2) -1, size(ID_lib.ID, 2)) + 1)],'fontsize',12,'Color','c');
%             end
%         end
    end
end

function temp=generateTemplate(r)
    size_temp=2*r+1;
    template=ones(size_temp*4);
    template(1:size_temp*2+1,1:size_temp*2+1)=0;
    rr=round(size_temp*sqrt(2))-1;
    for i=1:4
        rot=imrotate(template,45+90*(i-1));
        temp(i,:,:)=imresize(rot(round(size(rot,1)/2)-rr:round(size(rot,1)/2)+rr,round(size(rot,2)/2)-rr:round(size(rot,2)/2)+rr),[size_temp, size_temp]);
    end
    
    template=ones(size_temp*4);
    template(1:size_temp*2+1,1:size_temp*2+1)=0;
    template=imresize(template,0.25);
    for i=5:8
        temp(i,:,:)=imrotate(template,90*(i-1));
    end
end

function img=adaptiveThreshold(img, r)
    cnt1=1;
    for i=1:r:size(img,1)
        cnt2=1;
        for j=1:r:size(img,2)
            max_value(cnt1,cnt2) = max(max(img(i:i+r-1,j:j+r-1)));
            min_value(cnt1,cnt2) = min(min(img(i:i+r-1,j:j+r-1)));
            cnt2=cnt2+1;
        end
        cnt1=cnt1+1;
    end
    for i=1:size(max_value,1)
        for j=1:size(max_value,2)
            max_value_mid(i,j)=max(max(max_value(max(1,i-1):min(i+1,end),max(1,j-1):min(j+1,end))));
            min_value_mid(i,j)=min(min(min_value(max(1,i-1):min(i+1,end),max(1,j-1):min(j+1,end))));
        end
    end
    max_value=max_value_mid;
    min_value=min_value_mid;
    for i=1:size(img,1)
        for j=1:size(img,2)
            if img(i,j)<(max_value(fix((i-1)/r)+1,fix((j-1)/r)+1)*0.5+min_value(fix((i-1)/r)+1,fix((j-1)/r)+1)*0.5) && min_value(fix((i-1)/r)+1,fix((j-1)/r)+1)<0.25
                img(i,j)=0;
            else
                img(i,j)=1;
            end
        end
    end
end

function feature=areaGrowth(binary)
    L = bwlabel(~binary);
    for i=1:max(max(L))
        [r, c] = find(L==i);
        feature{i} = [r c];
    end
    illegal=zeros(1,size(feature,2));
    for i=1:size(feature,2)
        if size(feature{i},1)<30 || size(feature{i},1)>2000
            illegal(i)=1;
        end
    end
    feature(illegal==1)=[];
end

function ptList_refine=cornerSubPix(Gx, Gy, ptList, length, width)
    global graph
    for it = 1 : size(ptList,1)
        for i=1:3
            r_refine = 4;
            sig = 5;
            w_x=(-r_refine:r_refine);
            w_y=(-r_refine:r_refine);
            [X,Y]=meshgrid(w_x,w_y);
            W=exp(-(X.^2+Y.^2)./sig.^2);
            im = round(ptList(it,1));
            in = round(ptList(it,2));

            [M,N] = ndgrid(im-r_refine:im+r_refine,in-r_refine:in+r_refine);
%             M=M.*W;
%             N=N.*W;
            Gm = imgaussfilt(Gy(im-r_refine:im+r_refine,in-r_refine:in+r_refine),1).*W;
            Gn = imgaussfilt(Gx(im-r_refine:im+r_refine,in-r_refine:in+r_refine),1).*W;
            G = [Gm(:),Gn(:)];
            p = sum([M(:),N(:)].*G,2);
            now_refine=(G\p)';
            if norm(ptList(it,:)-now_refine)<0.01
                ptList_refine(it,:) = now_refine;
                break;
            end
            if now_refine(1)<r_refine+2 || now_refine(2)<r_refine+2 || now_refine(1)>width-r_refine-2 || now_refine(2)>length-r_refine-2
                ptList_refine(it,:) = now_refine;
                break;
            end
            ptList(it,:) = now_refine; 
        end
        ptList_refine(it,:) = now_refine;
    end
    % 清除靠近边界的点
    % remove the detected points near boundaries
    illegal = (any(ptList_refine<r_refine+2,2) | ptList_refine(:,1)>width-r_refine-2 | ptList_refine(:,2)>length-r_refine-2);
    ptList_refine(illegal,:)=[];
    graph(illegal,:)=[];
    graph(:,illegal)=[];
end

function res=judgeEdge(a, b, G)
    threshold=0.3;
    n=min(15, round(norm(a-b)/2));
    x=round(linspace(a(1),b(1),n));
    y=round(linspace(a(2),b(2),n));
    for i=1:n
        data(i)=G(x(i),y(i));
    end
    if mean(data)>threshold && min(data)>threshold-0.2
        res=1;
    else
        res=0;
    end
end

function addEdge(a,b)
    global graph
    graph(a,b)=1;
    graph(b,a)=1;
end

function dfs(start,now,clus)
    global visited square graph
    if size(clus,2) == 4 && graph(start,now) && judgeParallelogram(clus)
        square = [square; clus];
        visited(clus)=1;
        return;
    end
    for i=1:size(graph,1)
        if graph(now,i) && ~visited(i)
            visited(i)=1;
            dfs(start,i,[clus i]);
            visited(i)=0;
            if ~isempty(clus)
                clus(end)=[];
            end
        end
    end
end

function res=judgeParallelogram(clus)
    global ptList_refine
    percent=0.05;
    dis(1)=norm(ptList_refine(clus(1),:)-ptList_refine(clus(2),:));
    dis(2)=norm(ptList_refine(clus(2),:)-ptList_refine(clus(3),:));
    dis(3)=norm(ptList_refine(clus(3),:)-ptList_refine(clus(4),:));
    dis(4)=norm(ptList_refine(clus(1),:)-ptList_refine(clus(4),:));
    for i=1:4
        dis_to_center(i)=norm(ptList_refine(clus(i),:)-mean(ptList_refine(clus,:)));     
    end
    if dis_to_center(1)-dis_to_center(3)<percent*(dis_to_center(1)+dis_to_center(3)) && dis_to_center(2)-dis_to_center(4)<percent*(dis_to_center(2)+dis_to_center(4))
        res=1;
    else
        res=0;
    end
end

function res=judgeFeature(square1, square2)
    global ptList_refine
    threshold1 = 5;
    
    dis1(1)=norm(ptList_refine(square1(1),:)-ptList_refine(square1(2),:));
    dis1(2)=norm(ptList_refine(square1(2),:)-ptList_refine(square1(3),:));
    dis1(3)=norm(ptList_refine(square1(3),:)-ptList_refine(square1(4),:));
    dis1(4)=norm(ptList_refine(square1(1),:)-ptList_refine(square1(4),:));
    
    dis2(1)=norm(ptList_refine(square2(1),:)-ptList_refine(square2(2),:));
    dis2(2)=norm(ptList_refine(square2(2),:)-ptList_refine(square2(3),:));
    dis2(3)=norm(ptList_refine(square2(3),:)-ptList_refine(square2(4),:));
    dis2(4)=norm(ptList_refine(square2(1),:)-ptList_refine(square2(4),:));
    
    if dis1(1)>dis1(2)
        angle1=rad2deg(atan((ptList_refine(square1(1),2)-ptList_refine(square1(2),2))/(ptList_refine(square1(1),1)-ptList_refine(square1(2),1))));
    else
        angle1=rad2deg(atan((ptList_refine(square1(2),2)-ptList_refine(square1(3),2))/(ptList_refine(square1(2),1)-ptList_refine(square1(3),1))));
    end
    if dis2(1)>dis2(2)
        angle2=rad2deg(atan((ptList_refine(square2(1),2)-ptList_refine(square2(2),2))/(ptList_refine(square2(1),1)-ptList_refine(square2(2),1))));
    else
        angle2=rad2deg(atan((ptList_refine(square2(2),2)-ptList_refine(square2(3),2))/(ptList_refine(square2(2),1)-ptList_refine(square2(3),1))));
    end
    
    if (abs(rad2deg(atan((mean(ptList_refine(square1,2))-mean(ptList_refine(square2,2)))/(mean(ptList_refine(square1,1))-mean(ptList_refine(square2,1))))) - angle1) < threshold1 ...
            || abs(abs(rad2deg(atan((mean(ptList_refine(square1,2))-mean(ptList_refine(square2,2)))/(mean(ptList_refine(square1,1))-mean(ptList_refine(square2,1))))) - angle1) - 90) < threshold1)...
        && (abs(rad2deg(atan((mean(ptList_refine(square1,2))-mean(ptList_refine(square2,2)))/(mean(ptList_refine(square1,1))-mean(ptList_refine(square2,1))))) - angle2) < threshold1 ...
            || abs(abs(rad2deg(atan((mean(ptList_refine(square1,2))-mean(ptList_refine(square2,2)))/(mean(ptList_refine(square1,1))-mean(ptList_refine(square2,1))))) - angle2) - 90) < threshold1) ...
        && min(abs(rad2deg(atan((mean(ptList_refine(square1,2))-mean(ptList_refine(square2,2)))/(mean(ptList_refine(square1,1))-mean(ptList_refine(square2,1))))) - angle1), abs(rad2deg(atan((mean(ptList_refine(square1,2))-mean(ptList_refine(square2,2)))/(mean(ptList_refine(square1,1))-mean(ptList_refine(square2,1))))) - angle2)) < threshold1 ...
        && abs(norm(mean(ptList_refine(square1,:))-mean(ptList_refine(square2,:))) - (max(dis1)+max(dis2))/2) < 4 * min(min(dis1),min(dis2)) ...
        && abs(min(dis1)-min(dis2)) < min(min(dis1),min(dis2)) * 0.2 ...
        && (max(dis1) + max(dis2)) > 2 * (min(dis1) + min(dis2))
        res = 1;
    else
        res = 0;
        return;
    end
end

function feature=organizeFeature(square1, square2)
    global ptList_refine
    
    % ordering feature points 
    angle1=rad2deg(atan2(mean(ptList_refine(square1,1))-ptList_refine(square1,1),mean(ptList_refine(square1,2))-ptList_refine(square1,2)));
    angle2=rad2deg(atan2(mean(ptList_refine(square2,1))-ptList_refine(square2,1),mean(ptList_refine(square2,2))-ptList_refine(square2,2)));
    angle_middle=rad2deg(atan2(mean(ptList_refine(square2,1))-mean(ptList_refine(square1,1)),mean(ptList_refine(square2,2))-mean(ptList_refine(square1,2))));
    
    pos1=1;
    pos2=1;
    angle_min=360;
    angle_max=0;
    for i=1:4
        if abs(angle1(mod(i+1,4)+1)-angle_middle)+abs(angle1(mod(i+2,4)+1)-angle_middle)>angle_max
            pos1=i;
            angle_max=abs(angle1(mod(i+1,4)+1)-angle_middle)+abs(angle1(mod(i+2,4)+1)-angle_middle);
        end
        if abs(angle2(mod(i+1,4)+1)-angle_middle)+abs(angle2(mod(i+2,4)+1)-angle_middle)<angle_min
            pos2=i;
            angle_min=abs(angle2(mod(i+1,4)+1)-angle_middle)+abs(angle2(mod(i+2,4)+1)-angle_middle);
        end
    end
    square1=[square1(pos1:end) square1(1:pos1-1)];
    square2=[square2(pos2:end) square2(1:pos2-1)];
    
    if angle1(pos1)<angle1(mod(pos1,4)+1) && abs(angle1(pos1)-angle1(mod(pos1,4)+1)) < 90
        square1=square1([2 1 4 3]);
    end
    if angle1(pos1)>angle1(mod(pos1,4)+1) && abs(angle1(pos1)-angle1(mod(pos1,4)+1)) > 270
        square1=square1([2 1 4 3]);
    end
    if angle2(pos2)<angle2(mod(pos2,4)+1) && abs(angle2(pos2)-angle2(mod(pos2,4)+1)) < 90
        square2=square2([2 1 4 3]);
    end
    if angle2(pos2)>angle2(mod(pos2,4)+1) && abs(angle2(pos2)-angle2(mod(pos2,4)+1)) > 270
        square2=square2([2 1 4 3]);
    end
    
    feature=[square1,square2];
end

function [ID, center, direction] = extractFeature(img,feature)
    decoder=[1.45,0;1.54,0;1.63,0;1.72,0;1.8,0;1.72,1;1.63,1;1.54,1;1.45,1];
    global ptList_refine
    ID=zeros(1, size(feature,1));
    center=[];
    direction=[];
    for i=1:size(feature,1)
        area1 = polyarea(ptList_refine(feature(i,1:4),1),ptList_refine(feature(i,1:4),2));
        area2 = polyarea(ptList_refine(feature(i,5:8),1),ptList_refine(feature(i,5:8),2));
        area_middle = polyarea(ptList_refine(feature(i,[3 4 7 8]),1),ptList_refine(feature(i,[3 4 7 8]),2));
        area_all = polyarea(ptList_refine(feature(i,[1 2 5 6]),1),ptList_refine(feature(i,[1 2 5 6]),2));
%         area_all = area1 + area2 + area_middle;
        cross_ratio = ((area1+area_middle)*(area2+area_middle))/(area_all * area_middle);
        if area1>area2
            label1=0;
        else
            label1=1;
        end
        if img(round(mean(ptList_refine(feature(i,1:4),1))),round(mean(ptList_refine(feature(i,1:4),2))))>img(round(mean(ptList_refine(feature(i,5:8),1))),round(mean(ptList_refine(feature(i,5:8),2))))
            label2=0;
        else
            label2=1;
        end
        if ~xor(label1,label2)
            label=0;
        end
        if xor(label1,label2)
            label=1;
        end
        diff=5;
        pos=-1;
        for it=1:9
            if diff>abs(decoder(it,1)-cross_ratio) && it==5
                diff=abs(decoder(it,1)-cross_ratio);
                pos=it;
            end
            if diff>abs(decoder(it,1)-cross_ratio) && label==decoder(it,2)
                diff=abs(decoder(it,1)-cross_ratio);
                pos=it;
            end
        end
        if diff < 0.1
            ID(i)=cross_ratio;
            center(i,:)=mean(ptList_refine(feature(i,[1 2 5 6]),:));
            direction(i)=rad2deg(atan2(mean(ptList_refine(feature(i,1:4),1))-mean(ptList_refine(feature(i,5:8),1)),mean(ptList_refine(feature(i,1:4),2))-mean(ptList_refine(feature(i,5:8),2))));
        end
    end
end

function [ID_num, ID_pos, direc]=findID(ID,ID_lib)
    maxSimilar=-1;
    ID_num=0;
    direc=0;
    ID_pos=0;
    conflict=0;
    for i=1:size(ID_lib,1)
        for j=1:size(ID,2)
            similar=judgeID(ID_lib(i,[j:end 1:j-1]),ID);
            if similar>maxSimilar
                maxSimilar=similar;
                ID_num=i;
                ID_pos=j;
                direc=1;
                conflict=0;
            elseif similar==maxSimilar
                conflict=1;
            end
            similar=judgeID(ID_lib(i,[j:-1:1 end:-1:j+1]),ID);
            if similar>maxSimilar
                maxSimilar=similar;
                ID_num=i;
                ID_pos=j;
                direc=-1;
                conflict=0;
            elseif similar==maxSimilar
                conflict=1;
            end
        end
    end
    if conflict
        ID_num=-1;
    end
end

function similar=judgeID(ID1, ID2)
    cnt=0;
    for i=1:size(ID1, 2)
        if ID1(i)==ID2(i)
            cnt=cnt+1;
        end
    end
    similar=cnt;
end