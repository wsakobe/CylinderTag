clc,clear;
global ID tag_length

tag_length = 1200;
ID=[];

CylinderTagGenerator(18, 4, 100, 15);

function max_number=select(feature_size)
    global judge_usage
    max_number=0;
    for i = 1:size(judge_usage,1)
        code = i - 1;
        judge_usage(i) = 1;
        illegal = 0;
        for j=1:feature_size
            a(j) = mod(code, 64);
            if (((mod(a(j),8)<=3 && fix(a(j)/8)>=4) || (mod(a(j),8)>=4 && fix(a(j)/8)<=3)))
                illegal = 1;
                break;
            end
            code = fix(code/64);
        end
        if illegal
            continue;
        end
        if ~illegal && i~=inverse(i,feature_size)
            judge_usage(i) = 0;
            max_number=max_number+1;
        end
    end
end

function CylinderTagGenerator(tag_col, feature_size, tag_number, height_width_ratio)
    global flag judge_usage ID max_tag_number 
 
    judge_usage=zeros(64^feature_size,1);
    max_tag_number=select(feature_size);
    max_tag_number=fix(max_tag_number/(2*tag_col));
    if tag_number > max_tag_number
        tag_number = max_tag_number;
        disp(['The size of CylinderTag is ', num2str(tag_number)]);
    end
    tic
    while size(ID,1) < tag_number
        if toc > 20
            break;
        end
        flag=0;
        dfs(tag_col,[],feature_size,tag_number);        
    end
    if testConflict(ID, feature_size)
        disp('The Marker Codes are unique!');
        plot_tag(ID, height_width_ratio);
        save ./CTag_Generated/ID.mat ID 
    else
        disp('The Marker Codes are not unique!');
    end
end

function dfs(col, ID_now, feature_size, tag_number)
    global flag ID judge_usage
    
    %% 搜索可行解成功，打标记
    if size(ID_now, 2) == col
        flag = 1;
        ID=[ID; ID_now];
        toc
        disp(['Now: ' num2str(size(ID,1)) '/' num2str(tag_number)]);
        return;
    end

    if size(ID_now, 2) < col - 1
        if isempty(ID_now)
            code_init = fix(rand*(64^feature_size))+1;
            while judge_usage(code_init)
                code_init=fix(rand*(64^feature_size))+1;
            end
            judge_usage(code_init)=1;
            judge_usage(inverse(code_init,feature_size))=1;
            code_initi = code_init;
            code_init = code_init - 1;
            for j=1:feature_size
                a(j) = mod(code_init, 64);
                code_init = fix(code_init/64);
            end
            dfs(col, a, feature_size, tag_number);
            if flag
                return;
            end
            judge_usage(code_initi)=0;
            judge_usage(inverse(code_initi,feature_size))=0;
        else
            waitlist=[];
            for i = 0:63
                if (mod(i,8)<=3 && fix(i/8)>=4) || (mod(i,8)>=4 && fix(i/8)<=3)
                    continue;
                end
                now = 1;
                for iter=1:feature_size-1
                    now = now + ID_now(end - feature_size + 1 + iter) * 64 ^ (iter - 1);
                end
                now = now + i * 64 ^ (feature_size - 1); 
                if ~judge_usage(now) && ~judge_usage(inverse(now,feature_size))
                    waitlist=[waitlist i];
                end
            end
            if isempty(waitlist)
                return;
            end
            choice_base = zeros(1, size(waitlist, 2));
            for i = 1:size(waitlist, 2)
                for j = 0:63
                    if (mod(j,8)<=3 && fix(j/8)>=4) || (mod(j,8)>=4 && fix(j/8)<=3)
                        continue;
                    end
                    now = 1;
                    if feature_size > 2
                        for iter=1:feature_size-2
                            now = now + ID_now(end - feature_size + 2 + iter) * 64 ^ (iter - 1);
                        end
                    else
                        now = now + waitlist(i);
                    end
                    now = now + j * 64 ^ (feature_size - 1);
                    if ~judge_usage(now) && ~judge_usage(inverse(now,feature_size))
                        choice_base(i) = choice_base(i) + 1;
                    end
                end
            end
            [choice_base, p] = sort(choice_base, 'descend');
            equal_elements = find(diff(choice_base)~=0);
            equal_elements = [equal_elements size(choice_base, 2)];
            start=1;
            for ii = equal_elements
                idx = start:ii;
                p(idx) = p(idx(randperm(numel(idx))));
                start=ii+1;
            end
            for i=1:size(p,2)
                if choice_base(i)==0
                    break;
                end
                next_step = waitlist(p(i));
                next_score = 1;
                for iter = 1:feature_size - 1
                    next_score = next_score + ID_now(end - feature_size + 1 + iter) * 64 ^ (iter - 1);
                end
                next_score = next_score + next_step * 64 ^ (feature_size - 1);
                judge_usage(next_score) = 1;
                judge_usage(inverse(next_score,feature_size)) = 1;
                dfs(col, [ID_now next_step], feature_size, tag_number);
                if flag
                    return;
                end
                judge_usage(next_score) = 0;
                judge_usage(inverse(next_score,feature_size)) = 0;
            end
        end
    end
    if size(ID_now, 2) == col - 1
        for it=1:100
            i=fix(rand*64);
            while (mod(i,8)<=3 && fix(i/8)>=4) || (mod(i,8)>=4 && fix(i/8)<=3)
                i=fix(rand*64);
            end
            for iter=1:feature_size
                now_cyc(iter) = 1;
                for j=1:feature_size
                    if  (iter + j - feature_size) == 1
                        now_cyc(iter) = now_cyc(iter) + i * 64 ^ (j - 1);
                    elseif (iter + j - feature_size) < 1
                        now_cyc(iter) = now_cyc(iter) + ID_now(end - feature_size + iter + j) * 64 ^ (j - 1);
                    elseif (iter + j - feature_size) > 1
                        now_cyc(iter) = now_cyc(iter) + ID_now(-feature_size + iter + j - 1) * 64 ^ (j - 1);
                    end
                end
            end
            if ~sum(judge_usage(now_cyc)) && ~sum(judge_usage(inverse(now_cyc,feature_size))) && isempty(intersect(now_cyc,inverse(now_cyc,feature_size)))
                judge_usage(now_cyc) = 1;
                judge_usage(inverse(now_cyc,feature_size)) = 1;
                dfs(col, [ID_now i],feature_size, tag_number);
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
        code = in(iter)-1;
        for j=1:feature_size
            a(j) = mod(code, 64);
            a1(j) = (7 - mod(a(j),8)) * 8 + (7 - fix(a(j)/8));
            code = fix(code/64);
        end
        out(iter) = 1;
        for i=1:feature_size
            out(iter) = out(iter) + a1(i) * 64 ^ (feature_size - i);
        end
    end
end

function plot_tag(ID, ratio)
    global tag_length
    
    for i=1:size(ID,1)
        background=ones(tag_length,1.5*tag_length/ratio*size(ID,2));
        for j=1:size(ID,2)
            background=draw(background,j-1,ID(i,j),ratio);
        end
        imshow(background)
        imwrite(background,['./CTag_Generated/cy' num2str(i) '.bmp'])
    end
end

function background=draw(background, cnt, ID_now, ratio)
    global tag_length
    decoder=[1.47,0;1.54,0;1.61,0;1.68,0;1.68,1;1.61,1;1.54,1;1.47,1];
    white_ratio=0.2;
    left=fix(ID_now/8)+1;
    right=mod(ID_now,8)+1;
    block_pos_left=roots([-1 tag_length (white_ratio/2+white_ratio*white_ratio/4-0.2*decoder(left,1))*tag_length*tag_length]);
    block_pos_left=block_pos_left((block_pos_left>0));
    block_pos_left=block_pos_left((block_pos_left<tag_length*(1-white_ratio)));
    if decoder(left,2)
        block_pos_left = max(block_pos_left);
    else
        block_pos_left = min(block_pos_left);
    end
    block_pos_right=roots([-1 tag_length (white_ratio/2+white_ratio*white_ratio/4-0.2*decoder(right,1))*tag_length*tag_length]);
    block_pos_right=block_pos_right((block_pos_right>0));
    block_pos_right=block_pos_right((block_pos_right<tag_length*(1-white_ratio)));
    if decoder(right,2)
        block_pos_right = max(block_pos_right);
    else
        block_pos_right = min(block_pos_right);
    end
    background = insertShape(background,'FilledPolygon',[tag_length/ratio * 1.5 * cnt 0; tag_length/ratio * 1.5 * cnt block_pos_left-tag_length*white_ratio/2; tag_length/ratio * 1.5 * cnt + tag_length/ratio block_pos_right-tag_length*white_ratio/2; tag_length/ratio * 1.5 * cnt + tag_length/ratio 0],'Color','black','Opacity',1,'SmoothEdges', false);
    background = insertShape(background,'FilledPolygon',[tag_length/ratio * 1.5 * cnt tag_length; tag_length/ratio * 1.5 * cnt block_pos_left + tag_length*white_ratio/2; tag_length/ratio * 1.5 * cnt + tag_length/ratio block_pos_right + tag_length*white_ratio/2; tag_length/ratio * 1.5 * cnt + tag_length/ratio tag_length],'Color','black','Opacity',1,'SmoothEdges', false);
end

function res=testConflict(Code,feature_size)
    judge_conf=zeros(64^feature_size,1);
    for i=1:size(Code,1)
        for j=1:size(Code,2)
            score=1;
            for k=1:feature_size
                score=score+Code(i,mod(j+k-2,size(Code,2))+1)*64^(k-1);
            end
            if judge_conf(score)
                res = 0;
                return;
            else
                judge_conf(score)=1;
            end
        end
    end
    Code1 = Code;
    Code=fliplr(Code);
    for i=1:size(Code,1)
        for j=1:size(Code,2)
            Code(i,j)=(7 - mod(Code(i,j),8)) * 8 + (7 - fix(Code(i,j)/8));
        end
    end
    
    for i=1:size(Code,1)
        for j=1:size(Code,2)
            score=1;
            for k=1:feature_size
                score=score+Code(i,mod(j+k-2,size(Code,2))+1)*64^(k-1);
            end
            if judge_conf(score)
                res = 0;
                return;
            else
                judge_conf(score)=1;
            end
        end
    end
    res = 1;
end