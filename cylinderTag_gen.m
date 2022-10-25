clc,clear;
global ID tag_length 

tag_length=1500;
ID=[];
decoder=[1.45,0;1.54,0;1.63,0;1.72,0;1.8,0;1.72,1;1.63,1;1.54,1;1.45,1];

CylinderTagGenerator(15, 15, 4);

plot_tag(ID);

save ./CTag_Generated/ID.mat ID 

function select(feature_size)
    global judge_usage
    for i = 1:size(judge_usage,1)
        code = i - 1;
        for j=1:feature_size
            a(j) = mod(code, 9);
            code = fix(code/9);
        end
        judge_usage(i) = 1;
        for j=1:feature_size
            if a(j) ~= a(feature_size-j+1)
                judge_usage(i) = 0;
                break;
            end
        end
    end
end

function CylinderTagGenerator(tag_number, tag_col, feature_size)
    global flag judge_usage
    judge_usage=zeros(9^feature_size,1);
    select(feature_size);
    for i=1:tag_number
        flag=0;
        dfs(tag_col,[],feature_size);        
    end
end

function dfs(col, ID_now, feature_size)
    global flag ID judge_usage
    
    %% 搜索可行解成功，打标记
    if size(ID_now, 2) == col
        flag = 1;
        ID=[ID; ID_now];
        size(ID,1)
        return;
    end

    if size(ID_now, 2) < col - 1
        if isempty(ID_now)
            for i=0:8
                dfs(col, [ID_now i], feature_size);
                if flag
                    return;
                end
            end
        else
            for i=[mod(ID_now(end)+4,9):8 0:mod(ID_now(end)+4,9)]
                if size(ID_now, 2) < feature_size - 1
                    dfs(col, [ID_now i], feature_size);
                    if flag
                        return;
                    end
                else    
                    now = 1;
                    for iter=1:feature_size-1
                        now = now + ID_now(end - feature_size + 1 + iter) * 9 ^ (iter - 1);
                    end
                    now = now + i * 9 ^ (feature_size - 1); 
                    if (~judge_usage(now) && ~judge_usage(inverse(now,feature_size)))
                        judge_usage(now) = 1;
                        judge_usage(inverse(now,feature_size)) = 1;
                        dfs(col, [ID_now i], feature_size);
                        if flag
                            return;
                        end
                        judge_usage(now) = 0;
                        judge_usage(inverse(now,feature_size)) = 0;
                    end
                end
            end
        end
    end
    if size(ID_now, 2) == col - 1
        for i=0:8
            for iter=1:feature_size
                now_cyc(iter) = 1;
                for j=1:feature_size
                    if  (iter + j - feature_size) == 1
                        now_cyc(iter) = now_cyc(iter) + i * 9 ^ (j - 1);
                    elseif (iter + j - feature_size) < 1
                        now_cyc(iter) = now_cyc(iter) + ID_now(end - feature_size + iter + j) * 9 ^ (j - 1);
                    elseif (iter + j - feature_size) > 1
                        now_cyc(iter) = now_cyc(iter) + ID_now(-feature_size + iter + j - 1) * 9 ^ (j - 1);
                    end
                end
            end
            if ~sum(judge_usage(now_cyc)) && ~sum(judge_usage(inverse(now_cyc,feature_size)))
                judge_usage(now_cyc) = 1;
                judge_usage(inverse(now_cyc,feature_size)) = 1;
                dfs(col, [ID_now i],feature_size);
                if flag
                    return;
                end
                judge_usage(now_cyc) = 0;
                judge_usage(inverse(now_cyc,feature_size)) = 0;
            end
        end
    end
end
    
function out = inverse(in,feature_size)
    for iter=1:size(in,2)
        code = in(iter) - 1;
        for j=1:feature_size
            a(j) = mod(code, 9);
            code = fix(code/9);
        end
        out(iter) = 1;
        for i=1:feature_size
            out(iter) = out(iter) + a(i) * 9 ^ (feature_size - i);
        end
    end
end

function plot_tag(ID)
    global tag_length
    for i=1:size(ID,1)
        background=ones(tag_length,2*tag_length/10*size(ID,2));
        for j=1:size(ID,2)
            background=draw(background,j,ID(i,j)+1);
        end
        imshow(background)
%         imwrite(background,['./CTag_Generated/cy' num2str(i) '.bmp'])
    end
end

function background=draw(background, cnt, ID_now)
    global tag_length
    decoder=[1.45,0;1.54,0;1.63,0;1.72,0;1.8,0;1.72,1;1.63,1;1.54,1;1.45,1];
    temp_feature=zeros(tag_length,tag_length/10);
    block_pos=roots([-1 4*tag_length/5 tag_length*tag_length/5*(1-decoder(ID_now,1))]);
    block_pos=block_pos((block_pos>0));
    block_pos=block_pos((block_pos<tag_length*4/5));
    if decoder(ID_now,2)
        block_pos=max(block_pos);
    else
        block_pos=min(block_pos);
    end
    temp_feature(round(block_pos)+1:round(block_pos)+tag_length/5,:)=ones(tag_length/5,tag_length/10);
    temp_feature(round(block_pos)+tag_length/5+1:end,:)=temp_feature(round(block_pos)+tag_length/5+1:end,:)+0.25;
    background(:,(cnt-1)*2*tag_length/10+1:cnt*2*tag_length/10)=[temp_feature ones(tag_length,tag_length/10)];
end