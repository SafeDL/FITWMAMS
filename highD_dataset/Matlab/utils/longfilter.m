function [tracks_filter,tracks_temp,Result] = longfilter(tracks)
%% 只提取跟驰数据，对于换道数据或者巡航数据剔除
i = 1;
index = zeros(length(tracks),1);
for iRow  = 1:length(tracks)
   if ((tracks(iRow).numLaneChanges ~= 0) || (tracks(iRow).minDHW == -1) || strcmp(tracks(iRow).class,'Truck'))
        index(i) = iRow;
        i = i + 1;
   else
       continue;
   end    
end
% 删除它们
NumMax  = find(index > 0); % 找到原始样本自由巡航、换道、以及跟随卡车行驶的index
tracks_filter = tracks;
tracks_filter(index(1:length(NumMax))) = [];
tracks_temp = tracks_filter;
% 在剩下的数据中只提取跟车距离在0~150之间的样本,大于150的样本数据全部删除
i = 1;
iindex = zeros(length(tracks_filter),1);
for iRow  = 1:length(tracks_filter)
    index = find((tracks_temp(iRow).dhw > 0) & (tracks_temp(iRow).dhw < 150));
    if (~isempty(index)) % 这条数据存在有效跟车
        tracks_temp(iRow).frames = tracks_temp(iRow).frames(index);
        tracks_temp(iRow).bbox = tracks_temp(iRow).bbox(index);
        tracks_temp(iRow).xVelocity = tracks_temp(iRow).xVelocity(index);
        tracks_temp(iRow).yVelocity = tracks_temp(iRow).yVelocity(index);
        tracks_temp(iRow).xAcceleration = tracks_temp(iRow).xAcceleration(index);
        tracks_temp(iRow).yAcceleration = tracks_temp(iRow).yAcceleration(index);
        tracks_temp(iRow).lane = tracks_temp(iRow).lane(index);
        tracks_temp(iRow).thw = tracks_temp(iRow).thw(index);
        tracks_temp(iRow).ttc = tracks_temp(iRow).ttc(index);
        tracks_temp(iRow).dhw = tracks_temp(iRow).dhw(index);
        tracks_temp(iRow).frontSightDistance = tracks_temp(iRow).frontSightDistance(index);
        tracks_temp(iRow).backSightDistance = tracks_temp(iRow).backSightDistance(index);
        tracks_temp(iRow).precedingXVelocity = tracks_temp(iRow).precedingXVelocity(index);
        tracks_temp(iRow).precedingId = tracks_temp(iRow).precedingId(index);
        tracks_temp(iRow).followingId = tracks_temp(iRow).followingId(index);
        tracks_temp(iRow).leftPrecedingId = tracks_temp(iRow).leftPrecedingId(index);
        tracks_temp(iRow).leftAlongsideId = tracks_temp(iRow).leftAlongsideId(index);
        tracks_temp(iRow).leftFollowingId = tracks_temp(iRow).leftFollowingId(index);
        tracks_temp(iRow).rightPrecedingId = tracks_temp(iRow).rightPrecedingId(index);
        tracks_temp(iRow).rightAlongsideId = tracks_temp(iRow).rightAlongsideId(index);
        tracks_temp(iRow).rightFollowingId = tracks_temp(iRow).rightFollowingId(index);
    else % 非有效跟车
        iindex(i) = iRow;
        i = i+1;
    end 
end
% 删除无效跟车数据，序列存储在iindex中
NumofInvalid  = find(iindex > 0); 
tracks_temp(iindex(NumofInvalid)) = [];
%% 提取关键指标参数存为结构体
% Initialize tracks 
Result = {}; % 返回的是1*n的结构体
% Initialize constant variables
trackIndex = 1;    
% Iterate over the whole table
for iRow = 1:size(tracks_temp,2)
% for iRow = 97:97
    disp('iRow'); % 调试用
    disp(iRow); % 调试用
    iTrack = tracks_temp(iRow);% 临时变量
    % 根据先验知识，筛选0s < TTC < 10 s 的数据为有效跟随事件
    
    if ~isempty(find((iTrack.ttc > 10) | (iTrack.ttc <= 0))) % 找到不符合要求的索引并删除
        iindex = find((iTrack.ttc > 10) | (iTrack.ttc <= 0));

%     if ~isempty(iTrack.ttc <= 0) % 找到不符合要求的索引并删除
%         iindex = find(iTrack.ttc <= 0);
        
        iTrack.frames(iindex) = [];
        iTrack.bbox(iindex) = [];
        iTrack.xVelocity(iindex) = [];
        iTrack.yVelocity(iindex) = [];
        iTrack.xAcceleration(iindex) = [];
        iTrack.yAcceleration(iindex) = [];
        iTrack.lane(iindex) = [];
        iTrack.thw(iindex) = [];
        iTrack.ttc(iindex) = [];
        iTrack.dhw(iindex) = [];
        iTrack.frontSightDistance(iindex) = [];
        iTrack.backSightDistance(iindex) = [];
        iTrack.precedingXVelocity(iindex) = [];
        iTrack.precedingId(iindex) = [];
        iTrack.followingId(iindex) = [];
        iTrack.leftPrecedingId(iindex) = [];
        iTrack.leftAlongsideId(iindex) = [];
        iTrack.leftFollowingId(iindex) = [];
        iTrack.rightPrecedingId(iindex) = [];
        iTrack.rightAlongsideId(iindex) = [];
        iTrack.rightFollowingId(iindex) = [];
    else
        % do nothing
    end
    % 有效跟车点数小于10，剔除
    temp_precedingId = unique(iTrack.precedingId);
    for i = 1:length(temp_precedingId)
        if length(find((iTrack.precedingId == temp_precedingId(i)))) <= 10 
            iindex = find((iTrack.precedingId == temp_precedingId(i)));
            iTrack.frames(iindex) = [];
            iTrack.bbox(iindex) = [];
            iTrack.xVelocity(iindex) = [];
            iTrack.yVelocity(iindex) = [];
            iTrack.xAcceleration(iindex) = [];
            iTrack.yAcceleration(iindex) = [];
            iTrack.lane(iindex) = [];
            iTrack.thw(iindex) = [];
            iTrack.ttc(iindex) = [];
            iTrack.dhw(iindex) = [];
            iTrack.frontSightDistance(iindex) = [];
            iTrack.backSightDistance(iindex) = [];
            iTrack.precedingXVelocity(iindex) = [];
            iTrack.precedingId(iindex) = [];
            iTrack.followingId(iindex) = [];
            iTrack.leftPrecedingId(iindex) = [];
            iTrack.leftAlongsideId(iindex) = [];
            iTrack.leftFollowingId(iindex) = [];
            iTrack.rightPrecedingId(iindex) = [];
            iTrack.rightAlongsideId(iindex) = [];
            iTrack.rightFollowingId(iindex) = [];
            continue;
        else
            % do nothing
        end
    end
    % 如果发现剔除之后未发现有效跟车数据，跳过
    if isempty(iTrack.frames)
        continue;
    end
    Result(trackIndex).id = iTrack.id; % 提取车辆的ID号码
    Result(trackIndex).precedingId = unique(iTrack.precedingId); % 提取前车的ID号码
	% 自车速度信息
	Result(trackIndex).xVelocity = abs(iTrack.xVelocity);
    Result(trackIndex).maxVelocity = abs(max(iTrack.xVelocity));
    Result(trackIndex).minVelocity = abs(min(iTrack.xVelocity));
    Result(trackIndex).meanVelocity = abs(mean(iTrack.xVelocity));
    Result(trackIndex).stdVelocity = abs(std(iTrack.xVelocity));
	% 自车加速度信息
    xAcceleration = sign(iTrack.xVelocity(1))*iTrack.xAcceleration;
	Result(trackIndex).xAcceleration = xAcceleration;
    Result(trackIndex).maxxAcceleration = max(xAcceleration);
    Result(trackIndex).minxAcceleration = min(xAcceleration);
    Result(trackIndex).meanxAcceleration = mean(xAcceleration);
    Result(trackIndex).stdxAcceleration = std(xAcceleration);
	% 相对速度信息,前车 - 自车
    rVelocity = abs(iTrack.precedingXVelocity) - abs(iTrack.xVelocity);
	Result(trackIndex).rVelocity = rVelocity;
    Result(trackIndex).maxrVelocity = max(rVelocity);
	Result(trackIndex).minrVelocity = min(rVelocity);
	Result(trackIndex).meanrVelocity = mean(rVelocity);
	Result(trackIndex).stdrVelocity = std(rVelocity);
    % THW信息
	Result(trackIndex).THW = iTrack.thw;
	Result(trackIndex).minTHW = iTrack.minTHW;  
    % 跟车距离信息  
    Result(trackIndex).dhw = iTrack.dhw;
    Result(trackIndex).minDHW = iTrack.minDHW;
    Result(trackIndex).maxDHW = max(iTrack.dhw);  
    Result(trackIndex).meanDHW = mean(iTrack.dhw);
    Result(trackIndex).stdDHW = std(iTrack.dhw);
	% 前车速度信息
	Result(trackIndex).precedingXVelocity = abs(iTrack.precedingXVelocity);
    Result(trackIndex).maxprecedingXVelocity = abs(max(iTrack.precedingXVelocity));
    Result(trackIndex).minprecedingXVelocity = abs(min(iTrack.precedingXVelocity));
    Result(trackIndex).meanprecedingXVelocity = abs(mean(iTrack.precedingXVelocity));
    Result(trackIndex).stdprecedingXVelocity = abs(std(iTrack.precedingXVelocity));
    % TTC存储
    Result(trackIndex).TTC = iTrack.ttc;
    Result(trackIndex).minTTC = min(iTrack.ttc);
%% 减速和加速时间信息
    precedingId = unique(iTrack.precedingId);
    if length(precedingId) > 1 % 存在多个跟车对象
        preAx = []; % 前车加速度
        ax = []; % 临时存储加速度
        numofFrames = []; % 跟随状态时间帧
        dectime =  []; % 减速时间
        accetime = [];% 加速时间
        for i = 1:length(precedingId)
            % 判断行驶方向
            if tracks(precedingId(i)).drivingDirection == 1
                % 判断同一个跟车的ID是否分段,这里只考虑分成两段
                numofFrames = iTrack.frames(find(precedingId(i) == iTrack.precedingId));
                diffnumofFrames = diff(numofFrames);
                split_index = find(diffnumofFrames ~= 1); % 分段点的index
                if length(split_index) == 1 % 只分成2段
                    % 第一个子段
                    numofId_min = find(tracks(precedingId(i)).frames == numofFrames(1));
                    numofId_max = find(tracks(precedingId(i)).frames == numofFrames(split_index(1)));
                    ax = -1*tracks(precedingId(i)).xAcceleration(numofId_min:numofId_max);
                    preAx = [preAx;ax];
                    % 第二个子段
                    numofId_min = find(tracks(precedingId(i)).frames == numofFrames(split_index(1) + 1));
                    numofId_max = find(tracks(precedingId(i)).frames == numofFrames(end));
                    ax = -1*tracks(precedingId(i)).xAcceleration(numofId_min:numofId_max);
                    preAx = [preAx;ax];
                elseif isempty(split_index) % 未发生分段
                    numofId_min = find(tracks(precedingId(i)).frames == numofFrames(1));
                    numofId_max = find(tracks(precedingId(i)).frames == numofFrames(end));
                    ax = -1*tracks(precedingId(i)).xAcceleration(numofId_min:numofId_max);
                    preAx = [preAx;ax];
                else % 其他情况，分成3段或更多，过于复杂
                    % do nothing;
                end
                % 寻找减速对应过程
                index1 = find(ax < 0);
                % 寻找加速对应过程
                index2 = find(ax > 0);
                % 对index1和iindex2求差分，判别变速是否分段
                iindex1 = find(diff(index1) > 1); % 找到差分大于1的临界值
                if (~isempty(iindex1))
                    dec = zeros(length(iindex1) + 1,1);
                    dec(1) = (tracks(precedingId(i)).frames(index1(iindex1(1))) - tracks(precedingId(i)).frames(index1(1))) * 0.04; % 第一段减速时间
                    dec(end) = (tracks(precedingId(i)).frames(index1(end)) - tracks(precedingId(i)).frames(index1(iindex1(end)))) * 0.04; % 最后一段减速时间
                    % 中间的减速过程
                    for ii = 2:length(iindex1)
                        dec(ii) = (tracks(precedingId(i)).frames(index1(iindex1(ii))) - tracks(precedingId(i)).frames(index1(iindex1(ii-1)) + 1)) * 0.04;
                    end
                    dectime = [dectime;dec(:)];
                elseif (isempty(index1))
                    dectime = [dectime;[]];
                else % 只发生了一段减速过程
                    dectime = [dectime;(tracks(precedingId(i)).frames(index1(end)) - tracks(precedingId(i)).frames(index1(1))) * 0.04];
                end
                
                % 对index2求差分，判别加速是否分段
                iindex2 = find(diff(index2) > 1); % 找到差分大于1的临界值
                if (~isempty(iindex2))
                    acce = zeros(length(iindex2) + 1,1);
                    acce(1) = (tracks(precedingId(i)).frames(index2(iindex2(1))) - tracks(precedingId(i)).frames(index2(1))) * 0.04; % 第一段加速时间
                    acce(end) = (tracks(precedingId(i)).frames(index2(end)) - tracks(precedingId(i)).frames(index2(iindex2(end)))) * 0.04; % 最后一段加速时间
                    % 中间的加速过程
                    for ii = 2:length(iindex2)
                        acce(ii) = (tracks(precedingId(i)).frames(index2(iindex2(ii))) - tracks(precedingId(i)).frames(index2(iindex2(ii-1)) + 1)) * 0.04;
                    end
                    accetime = [accetime;acce(:)];
                elseif (isempty(index2))
                    accetime = [accetime;[]];
                else % 只发生了一段加速过程
                    accetime = [accetime;(tracks(precedingId(i)).frames(index2(end)) - tracks(precedingId(i)).frames(index2(1))) * 0.04];
                end
            else % 另一个行驶方向
                % 判断同一个跟车的ID是否分段,这里只考虑分成两段
                numofFrames = iTrack.frames(find(iTrack.precedingId == precedingId(i)));
                diffnumofFrames = diff(numofFrames);
                split_index = find(diffnumofFrames ~= 1); % 分段点的index
                if length(split_index) == 1 % 只分成2段
                    % 第一个子段
                    numofId_min = find(tracks(precedingId(i)).frames == numofFrames(1));
                    numofId_max = find(tracks(precedingId(i)).frames == numofFrames(split_index(1)));
                    ax = tracks(precedingId(i)).xAcceleration(numofId_min:numofId_max);
                    preAx = [preAx;ax];
                    % 第二个子段
                    numofId_min = find(tracks(precedingId(i)).frames == numofFrames(split_index(1) + 1));
                    numofId_max = find(tracks(precedingId(i)).frames == numofFrames(end));
                    ax = tracks(precedingId(i)).xAcceleration(numofId_min:numofId_max);
                    preAx = [preAx;ax];
                elseif isempty(split_index) % 未发生分段
                    numofId_min = find(tracks(precedingId(i)).frames == numofFrames(1));
                    numofId_max = find(tracks(precedingId(i)).frames == numofFrames(end));
                    ax = tracks(precedingId(i)).xAcceleration(numofId_min:numofId_max);
                    preAx = [preAx;ax];
                else % 其他情况，分成3段或更多，过于复杂剔除
                    % do nothing;
                end
                % 寻找减速对应帧
                index1 = find(ax < 0);
                % 寻找加速对应帧
                index2 = find(ax > 0);
                % 对index求差分，判别减速是否分段
                iindex1 = find(diff(index1) > 1); % 找到差分大于1的部分
                if (~isempty(iindex1)) % 发生了不止一段的减速过程
                    dec = zeros(length(iindex1) + 1,1);
                    % 分段存储
                    dec(1) = (tracks(precedingId(i)).frames(index1(iindex1(1))) - tracks(precedingId(i)).frames(index1(1))) * 0.04; % 第一段减速时间
                    dec(end) = (tracks(precedingId(i)).frames(index1(end)) - tracks(precedingId(i)).frames(index1(iindex1(end)))) * 0.04; % 最后一段减速时
                    % 中间的减速过程
                    for ii = 2:length(iindex1)
                        dec(ii) = (tracks(precedingId(i)).frames(index1(iindex1(ii))) - tracks(precedingId(i)).frames(index1(iindex1(ii-1)) + 1)) * 0.04;
                    end
                    dectime = [dectime;dec(:)];
                elseif (isempty(index1)) % 没有减速过程
                    dectime = [dectime;[]];
                else % 只发生了一段减速过程
                    dectime = [dectime;(tracks(precedingId(i)).frames(index1(end)) - tracks(precedingId(i)).frames(index1(1))) * 0.04];
                end
                
                % 对index2求差分，判别加速是否分段
                iindex2 = find(diff(index2) > 1); % 找到差分大于1的临界值
                if (~isempty(iindex2))
                    acce = zeros(length(iindex2) + 1,1);
                    acce(1) = (tracks(precedingId(i)).frames(index2(iindex2(1))) - tracks(precedingId(i)).frames(index2(1))) * 0.04; % 第一段加速时间
                    acce(end) = (tracks(precedingId(i)).frames(index2(end)) - tracks(precedingId(i)).frames(index2(iindex2(end)))) * 0.04; % 最后一段加速时间
                    % 中间的加速过程
                    for ii = 2:length(iindex2)
                        acce(ii) = (tracks(precedingId(i)).frames(index2(iindex2(ii))) - tracks(precedingId(i)).frames(index2(iindex2(ii-1)) + 1)) * 0.04;
                    end
                    accetime = [accetime;acce(:)];
                elseif (isempty(index2))
                    accetime = [accetime;[]];
                else % 只发生了一段加速过程
                    accetime = [accetime;(tracks(precedingId(i)).frames(index2(end)) - tracks(precedingId(i)).frames(index2(1))) * 0.04];
                end
            end
        end
        % 前车的加速度分布
        Result(trackIndex).preAx = preAx; 
        Result(trackIndex).maxpreAx = max(preAx);
        Result(trackIndex).minpreAx = min(preAx);
        Result(trackIndex).meanpreAx = mean(preAx);
        Result(trackIndex).stdpreAx = std(preAx);
        Result(trackIndex).dectime = dectime;
        Result(trackIndex).accetime = accetime;
    else % 只存在一个跟车对象
        % 初始化临时加速度存储空间
        preAx = []; % 前车加速度
        ax = []; % 临时存储加速度
        numofFrames = []; % 跟随状态时间帧
        dectime =  []; % 减速时间
        accetime =  []; % 加速时间
        % 判断行驶方向
        if tracks(precedingId).drivingDirection == 1
            % 判断同一个跟车的ID是否分段,这里只考虑分成两段
            numofFrames = iTrack.frames(find(precedingId == iTrack.precedingId));
            diffnumofFrames = diff(numofFrames);
            split_index = find(diffnumofFrames ~= 1); % 分段点的index
            if length(split_index) == 1 % 只分成2段
                % 第一个子段
                numofId_min = find(tracks(precedingId).frames == numofFrames(1));
                numofId_max = find(tracks(precedingId).frames == numofFrames(split_index(1)));
                ax = -1*tracks(precedingId).xAcceleration(numofId_min:numofId_max);
                preAx = [preAx;ax];
                % 第二个子段
                numofId_min = find(tracks(precedingId).frames == numofFrames(split_index(1) + 1));
                numofId_max = find(tracks(precedingId).frames == numofFrames(end));
                ax = -1*tracks(precedingId).xAcceleration(numofId_min:numofId_max);
                preAx = [preAx;ax];
            elseif isempty(split_index) % 未发生分段
                numofId_min = find(tracks(precedingId).frames == numofFrames(1));
                numofId_max = find(tracks(precedingId).frames == numofFrames(end));
                ax = -1*tracks(precedingId).xAcceleration(numofId_min:numofId_max);
                preAx = [preAx;ax];
            else % 其他情况，分成3段或更多，过于复杂剔除
                % do nothing;
            end
            % 寻找减速对应帧
            index1 = find(ax < 0);
            % 寻找加速对应帧
            index2 = find(ax > 0);
            % 对index1求差分，判别减速是否分段
            iindex1 = find(diff(index1) > 1); % 找到差分大于1的部分  
            if (~isempty(iindex1)) %发生了不止一段的减速过程
                dec = zeros(length(iindex1) + 1,1);
                % 分段存储
                dec(1) = (tracks(precedingId).frames(index1(iindex1(1))) - tracks(precedingId).frames(index1(1))) * 0.04; % 第一段减速时间   
                dec(end) = (tracks(precedingId).frames(index1(end)) - tracks(precedingId).frames(index1(iindex1(end)))) * 0.04; % 最后一段减速时间
                % 中间的减速过程
                for ii = 2:length(iindex1)
                    dec(ii) = (tracks(precedingId).frames(index1(iindex1(ii))) - tracks(precedingId).frames(index1(iindex1(ii-1)) + 1)) * 0.04;                     
                end
                dectime = [dectime;dec(:)];
            elseif (isempty(index1)) % 没有发生减速过程
                dectime = [dectime;[]];
            else % 只发生了一段减速过程
                dectime = [dectime;(tracks(precedingId).frames(index1(end)) - tracks(precedingId).frames(index1(1))) * 0.04]; 
            end
            
            % 对index2求差分，判别加速是否分段
            iindex2 = find(diff(index2) > 1); % 找到差分大于1的临界值
            if (~isempty(iindex2))
                acce = zeros(length(iindex2) + 1,1);
                acce(1) = (tracks(precedingId).frames(index2(iindex2(1))) - tracks(precedingId).frames(index2(1))) * 0.04; % 第一段加速时间                  
                acce(end) = (tracks(precedingId).frames(index2(end)) - tracks(precedingId).frames(index2(iindex2(end)))) * 0.04; % 最后一段加速时间                   
                % 中间的加速过程
                for ii = 2:length(iindex2)
                    acce(ii) = (tracks(precedingId).frames(index2(iindex2(ii))) - tracks(precedingId).frames(index2(iindex2(ii-1)) + 1)) * 0.04;     
                end
                accetime = [accetime;acce(:)];
            elseif (isempty(index2)) 
                accetime = [accetime;[]];
            else % 只发生了一段加速过程
                accetime = [accetime;(tracks(precedingId).frames(index2(end)) - tracks(precedingId).frames(index2(1))) * 0.04];  
            end            
        else % 判断行驶方向
            % 判断同一个跟车的ID是否分段,这里只考虑分成两段
            numofFrames = iTrack.frames(find(precedingId == iTrack.precedingId));
            diffnumofFrames = diff(numofFrames);
            split_index = find(diffnumofFrames ~= 1); % 分段点的index
            if length(split_index) == 1 % 只分成2段
                % 第一个子段
                numofId_min = find(tracks(precedingId).frames == numofFrames(1));
                numofId_max = find(tracks(precedingId).frames == numofFrames(split_index(1)));
                ax = tracks(precedingId).xAcceleration(numofId_min:numofId_max);
                preAx = [preAx;ax];
                % 第二个子段
                numofId_min = find(tracks(precedingId).frames == numofFrames(split_index(1) + 1));
                numofId_max = find(tracks(precedingId).frames == numofFrames(end));
                ax = tracks(precedingId).xAcceleration(numofId_min:numofId_max);
                preAx = [preAx;ax];
            elseif isempty(split_index) % 未发生分段
                numofId_min = find(tracks(precedingId).frames == numofFrames(1));
                numofId_max = find(tracks(precedingId).frames == numofFrames(end));
                ax = tracks(precedingId).xAcceleration(numofId_min:numofId_max);
                preAx = [preAx;ax];
            else % 其他情况，分成3段或更多，过于复杂剔除
                % do nothing;
            end
            % 寻找减速对应帧
            index1 = find(ax < 0);
            % 寻找加速对应帧
            index2 = find(ax > 0);
            % 对index1求差分，判别减速是否分段
            iindex1 = find(diff(index1) > 1); % 找到差分大于1的部分       
            if (~isempty(iindex1)) %发生了不止一段的减速过程
                dec = zeros(length(iindex1) + 1,1);
                % 分段存储
                dec(1) = (tracks(precedingId).frames(index1(iindex1(1))) - tracks(precedingId).frames(index1(1))) * 0.04; % 第一段减速时间   
                dec(end) = (tracks(precedingId).frames(index1(end)) - tracks(precedingId).frames(index1(iindex1(end)))) * 0.04; % 最后一段减速时间  
                % 中间的减速过程
                for ii = 2:length(iindex1)
                    dec(ii) = (tracks(precedingId).frames(index1(iindex1(ii))) - tracks(precedingId).frames(index1(iindex1(ii-1)) + 1)) * 0.04;                     
                end
                dectime = [dectime;dec(:)];
            elseif (isempty(index1)) % 没有发生减速过程
                dectime = [dectime;[]];
            else % 只发生了一段减速过程
                dectime = [dectime;(tracks(precedingId).frames(index1(end)) - tracks(precedingId).frames(index1(1))) * 0.04];
            end
            
            % 对index2求差分，判别加速是否分段
            iindex2 = find(diff(index2) > 1); % 找到差分大于1的临界值
            if (~isempty(iindex2))
                acce = zeros(length(iindex2) + 1,1);
                acce(1) = (tracks(precedingId).frames(index2(iindex2(1))) - tracks(precedingId).frames(index2(1))) * 0.04; % 第一段加速时间                  
                acce(end) = (tracks(precedingId).frames(index2(end)) - tracks(precedingId).frames(index2(iindex2(end)))) * 0.04; % 最后一段加速时间                   
                % 中间的加速过程
                for ii = 2:length(iindex2)
                    acce(ii) = (tracks(precedingId).frames(index2(iindex2(ii))) - tracks(precedingId).frames(index2(iindex2(ii-1)) + 1)) * 0.04;     
                end
                accetime = [accetime;acce(:)];
            elseif (isempty(index2)) 
                accetime = [accetime;[]];
            else % 只发生了一段加速过程
                accetime = [accetime;(tracks(precedingId).frames(index2(end)) - tracks(precedingId).frames(index2(1))) * 0.04];  
            end 
        end
        % 前车的加速度分布
        Result(trackIndex).preAx = preAx; 
        Result(trackIndex).maxpreAx = max(preAx);
        Result(trackIndex).minpreAx = min(preAx);
        Result(trackIndex).meanpreAx = mean(preAx);
        Result(trackIndex).stdpreAx = std(preAx);
        % 前车减速状态存储
        Result(trackIndex).dectime = dectime; 
        Result(trackIndex).accetime = accetime;
    end
    % Increment the internal track index 
    trackIndex = trackIndex + 1;
end
%% 对Result分析，剔除不存在加速度的值
iindex = [];
for i = 1:size(Result,2)
    temp = Result(i);
    if (length(temp.preAx) ~= length(temp.xVelocity))
        iindex = [iindex;i];
    end
end
Result(iindex) = [];
%% 找到需要显示的index
iindex = [];
for i = 1:size(Result,2)
    temp = Result(i);
	iindex = [iindex;temp.id;temp.precedingId];
end
unique_iindex = unique(iindex);
tracks_filter = tracks(unique_iindex);
end
