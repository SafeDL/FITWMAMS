function [tracks_filter,tracks_temp1,Result] = longfilter_onlycar(tracks)
%% 根据四个条件初筛跟随事件：
% 1：遍历车辆ID，根据是否存在PrecedingID获取跟随事件
% 2：如果跟车距离大于150m，删除
% 3：如果自车或跟随ID车辆存在换道行为，删除
% 4：连续超过25个有效记录值，为一个合理的跟随事件
% 5：对跟随事件分段，将加速和减速分开
%% step1：遍历车辆ID，根据是否存在PrecedingID获取跟随事件
tracks_temp1 = []; % 剔除自由巡航值tracks之后的值
for iRow  = 1:length(tracks)
    % 判断precedingId是否为0
    iTrack = tracks(iRow);
    precedingId = unique(iTrack.precedingId);
    if length(precedingId) == 1
        if precedingId == 0
            continue; % 该段不存在可以跟随的前车
        else
            tracks_temp1 = [tracks_temp1;iTrack];  % 此时tracks_temp为剔除自由巡航值
        end
    else
        tracks_temp1 = [tracks_temp1;iTrack];
    end
end

tracks_temp2 = []; % 提取有效跟驰ID的值
% 剔除tracks_temp1中的无跟车状态，并切分一段记录中的多目标跟车状态
for iRow  = 1:length(tracks_temp1)
    % 找到precedingId > 0的部分
    iTrack = tracks_temp1(iRow);
    precedingId = unique(iTrack.precedingId);
    precedingId_valid_index = find(precedingId > 0); % 有效跟车ID的位置，如果存储该位置的变量长度为1，则只有一个有效ID
    if length(precedingId_valid_index) == 1
        % 只有一个有效跟车ID,如果跟车距离大于150m，删除
        precedingId_position = find((iTrack.precedingId == precedingId(end)) & (iTrack.dhw < 150)); % 有效跟车位置的索引
        if isempty(precedingId_position)
            continue;
        end
        temp = savedata(iTrack,precedingId_position,precedingId(end));
        % 剔除不连续的帧
        temp_frame_len = find(diff(temp.frames) > 1);
        if (~isempty(temp_frame_len))
            continue;
        end
        tracks_temp2 = [tracks_temp2;temp];
    else
        % 存在多个跟车ID,如果跟车距离大于150m，删除
        for i = 1:length(precedingId_valid_index)
            precedingId_position = find((iTrack.precedingId == precedingId(precedingId_valid_index(i))) & (iTrack.dhw < 150)); % 有效跟车位置的索引
            if isempty(precedingId_position)
                continue;
            end
            temp = savedata(iTrack,precedingId_position,precedingId(precedingId_valid_index(i)));
            % 剔除不连续的帧
            temp_frame_len = find(diff(temp.frames) > 1);
            if (~isempty(temp_frame_len))
                continue;
            end
            tracks_temp2 = [tracks_temp2;temp];
        end
    end
end
%% step2：按照加速度进一步分隔驾驶事件，ax >= 0 的点全部取出
tracks_temp3 = []; % 按照前车加速度变化进一步切割的值
for iRow  = 1:length(tracks_temp2)
    %     disp(iRow)
    iTrack = tracks_temp2(iRow); % 当前查找到的记录数据事件
    precedingId = iTrack.precedingId; % 当前记录数据的跟驰前车ID
    
    % 在precedingID视角下的全局索引
    numofId_min = find(tracks(precedingId).frames == iTrack.frames(1));
    numofId_max = find(tracks(precedingId).frames == iTrack.frames(end));
    % 判断行驶方向
    if tracks(precedingId).drivingDirection == 1
        ax = -1*tracks(precedingId).xAcceleration(numofId_min:numofId_max);
        tracks(precedingId).xAcceleration = -1.*tracks(precedingId).xAcceleration;
    else
        ax = tracks(precedingId).xAcceleration(numofId_min:numofId_max);
    end
    
    % 寻找减速对应帧
    index1 = find(ax < 0) + numofId_min - 1;
    % 寻找加速对应帧
    index2 = find(ax >= 0) + numofId_min - 1;
    % 对index1求差分，判别减速是否分段
    iindex1 = find(diff(index1) > 1); % 找到差分大于1的部分
    iindex2 = find(diff(index2) > 1); % 找到差分大于1的临界值
    if (~isempty(iindex1)) % 发生了不止一段的减速过程
        % 第一段减速存储
        start_position = find(iTrack.frames == tracks(precedingId).frames(index1(1))); % 根据帧，找到iTrack中对应的索引
        end_position = find(iTrack.frames == tracks(precedingId).frames(index1(iindex1(1)))); % 根据帧，找到iTrack中对应的索引
        precedingId_position = start_position:end_position;
        temp = savedata(iTrack,precedingId_position,precedingId);
        % kesai及DRAC安全指标计算
        [kesai,DRAC] = SafetyIndicator(temp.dhw, temp.ttc ,temp.xVelocity, temp.precedingXVelocity);
        temp.kesai = kesai;
        temp.DRAC = DRAC;
        % 前车加速度及变速时间存储
        temp_ax = tracks(precedingId).xAcceleration(index1(1):index1(iindex1(1)));
        temp.preAx = temp_ax;
        temp.maxpreAx = max(temp.preAx);
        temp.minpreAx = min(temp.preAx);
        temp.meanpreAx = mean(temp.preAx);
        temp.stdpreAx = std(temp.preAx);
        temp.decetime = (index1(iindex1(1)) - index1(1)) * 0.04;
        temp.accetime = 0;
        tracks_temp3 = [tracks_temp3;temp];
        
        % 最后一段减速存储
        start_position = find(iTrack.frames == tracks(precedingId).frames(index1(iindex1(end) + 1)));
        end_position = find(iTrack.frames == tracks(precedingId).frames(index1(end)));
        precedingId_position = start_position:end_position;
        temp = savedata(iTrack,precedingId_position,precedingId);
        % kesai及DRAC安全指标计算
        [kesai,DRAC] = SafetyIndicator(temp.dhw, temp.ttc ,temp.xVelocity, temp.precedingXVelocity);
        temp.kesai = kesai;
        temp.DRAC = DRAC;
        % 前车加速度及变速时间存储
        temp_ax = tracks(precedingId).xAcceleration(index1(iindex1(end) + 1):index1(end));
        temp.preAx = temp_ax;
        temp.maxpreAx = max(temp.preAx);
        temp.minpreAx = min(temp.preAx);
        temp.meanpreAx = mean(temp.preAx);
        temp.stdpreAx = std(temp.preAx);
        temp.decetime = (index1(end) - index1(iindex1(end) + 1)) * 0.04;
        temp.accetime = 0;
        tracks_temp3 = [tracks_temp3;temp];
        
        % 中间的减速过程
        for ii = 2:length(iindex1)
            start_position = find(iTrack.frames == tracks(precedingId).frames(index1(iindex1(ii-1) + 1)));
            end_position = find(iTrack.frames == tracks(precedingId).frames(index1(iindex1(ii))));
            precedingId_position = start_position:end_position;
            temp = savedata(iTrack,precedingId_position,precedingId);
            % kesai及DRAC安全指标计算
            [kesai,DRAC] = SafetyIndicator(temp.dhw, temp.ttc ,temp.xVelocity, temp.precedingXVelocity);
            temp.kesai = kesai;
            temp.DRAC = DRAC;
            % 前车加速度及变速时间存储
            temp_ax = tracks(precedingId).xAcceleration(index1(iindex1(ii-1) + 1):index1(iindex1(ii)));
            temp.preAx = temp_ax;
            temp.maxpreAx = max(temp.preAx);
            temp.minpreAx = min(temp.preAx);
            temp.meanpreAx = mean(temp.preAx);
            temp.stdpreAx = std(temp.preAx);
            temp.decetime = (index1(iindex1(ii)) - index1(iindex1(ii-1) + 1)) * 0.04;
            temp.accetime = 0;
            tracks_temp3 = [tracks_temp3;temp];
        end
        
    % 只发生了一段减速过程
    elseif(~isempty(index1))
        start_position = find(iTrack.frames == tracks(precedingId).frames(index1(1)));
        end_position = find(iTrack.frames == tracks(precedingId).frames(index1(end)));
        precedingId_position = start_position:end_position;
        temp = savedata(iTrack,precedingId_position,precedingId);
        % kesai及DRAC安全指标计算
        [kesai,DRAC] = SafetyIndicator(temp.dhw, temp.ttc ,temp.xVelocity, temp.precedingXVelocity);
        temp.kesai = kesai;
        temp.DRAC = DRAC;
        % 前车加速度及变速时间存储
        temp_ax = tracks(precedingId).xAcceleration(index1(1):index1(end));
        temp.preAx = temp_ax;
        temp.maxpreAx = max(temp.preAx);
        temp.minpreAx = min(temp.preAx);
        temp.meanpreAx = mean(temp.preAx);
        temp.stdpreAx = std(temp.preAx);
        temp.decetime = (index1(end) - index1(1)) * 0.04;
        temp.accetime = 0;
        tracks_temp3 = [tracks_temp3;temp];
    end
    
    % 判别加速是否分段
    if (~isempty(iindex2))
        % 第一段加速存储
        start_position = find(iTrack.frames == tracks(precedingId).frames(index2(1)));
        end_position = find(iTrack.frames == tracks(precedingId).frames(index2(iindex2(1))));
        precedingId_position = start_position:end_position;
        temp = savedata(iTrack,precedingId_position,precedingId);
        % kesai及DRAC安全指标计算
        [kesai,DRAC] = SafetyIndicator(temp.dhw, temp.ttc ,temp.xVelocity, temp.precedingXVelocity);
        temp.kesai = kesai;
        temp.DRAC = DRAC;
        % 前车加速度及变速时间存储
        temp_ax = tracks(precedingId).xAcceleration(index2(1):index2(iindex2(1)));
        temp.preAx = temp_ax;
        temp.maxpreAx = max(temp.preAx);
        temp.minpreAx = min(temp.preAx);
        temp.meanpreAx = mean(temp.preAx);
        temp.stdpreAx = std(temp.preAx);
        temp.accetime = (index2(iindex2(1)) - index2(1)) * 0.04;
        temp.decetime = 0;
        tracks_temp3 = [tracks_temp3;temp];
        
        % 最后一段加速存储
        start_position = find(iTrack.frames == tracks(precedingId).frames(index2(iindex2(end) + 1)));
        end_position = find(iTrack.frames == tracks(precedingId).frames(index2(end)));
        precedingId_position = start_position:end_position;
        temp = savedata(iTrack,precedingId_position,precedingId);
        % kesai及DRAC安全指标计算
        [kesai,DRAC] = SafetyIndicator(temp.dhw, temp.ttc ,temp.xVelocity, temp.precedingXVelocity);
        temp.kesai = kesai;
        temp.DRAC = DRAC;
        % 前车加速度及变速时间存储
        temp_ax = tracks(precedingId).xAcceleration(index2(iindex2(end) + 1):index2(end));
        temp.preAx = temp_ax;
        temp.maxpreAx = max(temp.preAx);
        temp.minpreAx = min(temp.preAx);
        temp.meanpreAx = mean(temp.preAx);
        temp.stdpreAx = std(temp.preAx);
        temp.accetime = (index2(end) - index2(iindex2(end) + 1)) * 0.04;
        temp.decetime = 0;
        tracks_temp3 = [tracks_temp3;temp];
        
        % 中间过程的加速存储
        for ii = 2:length(iindex2)
            start_position = find(iTrack.frames == tracks(precedingId).frames(index2(iindex2(ii-1) + 1)));
            end_position = find(iTrack.frames == tracks(precedingId).frames(index2(iindex2(ii))));
            precedingId_position = start_position:end_position;
            temp = savedata(iTrack,precedingId_position,precedingId);
            % kesai及DRAC安全指标计算
            [kesai,DRAC] = SafetyIndicator(temp.dhw, temp.ttc ,temp.xVelocity, temp.precedingXVelocity);
            temp.kesai = kesai;
            temp.DRAC = DRAC;
            % 前车加速度及变速时间存储
            temp_ax = tracks(precedingId).xAcceleration(index2(iindex2(ii-1) + 1):index2(iindex2(ii)));
            temp.preAx = temp_ax;
            temp.maxpreAx = max(temp.preAx);
            temp.minpreAx = min(temp.preAx);
            temp.meanpreAx = mean(temp.preAx);
            temp.stdpreAx = std(temp.preAx);
            temp.accetime = (index2(iindex2(ii)) - index2(iindex2(ii-1) + 1)) * 0.04;
            temp.decetime = 0;
            tracks_temp3 = [tracks_temp3;temp];
        end
        
    % 只发生了一段加速过程
    elseif (~isempty(index2))
        start_position = find(iTrack.frames == tracks(precedingId).frames(index2(1)));
        end_position = find(iTrack.frames == tracks(precedingId).frames(index2(end)));
        precedingId_position = start_position:end_position;
        temp = savedata(iTrack,precedingId_position,precedingId);
        % kesai及DRAC安全指标计算
        [kesai,DRAC] = SafetyIndicator(temp.dhw, temp.ttc ,temp.xVelocity, temp.precedingXVelocity);
        temp.kesai = kesai;
        temp.DRAC = DRAC;
        % 前车加速度及变速时间存储
        temp_ax = tracks(precedingId).xAcceleration(index2(1):index2(end));
        temp.preAx = temp_ax;
        temp.maxpreAx = max(temp.preAx);
        temp.minpreAx = min(temp.preAx);
        temp.meanpreAx = mean(temp.preAx);
        temp.stdpreAx = std(temp.preAx);
        temp.accetime = (index2(end) - index2(1)) * 0.04;
        temp.decetime = 0;
        tracks_temp3 = [tracks_temp3;temp];
    end
end
%% -------------转存数据---------------%
    function temp = savedata(iTrack,precedingId_position,precedingId)
        temp.id = iTrack.id;
        temp.precedingId = precedingId;
        temp.frames = iTrack.frames(precedingId_position);
        % 自车速度信息
        temp.xVelocity = sign(iTrack.xVelocity(1)).*iTrack.xVelocity(precedingId_position);
        temp.maxVelocity = max(temp.xVelocity);
        temp.minVelocity = min(temp.xVelocity);
        temp.meanVelocity = mean(temp.xVelocity);
        temp.stdVelocity = std(temp.xVelocity);
        
        % 前车速度信息
        temp.precedingXVelocity = sign(iTrack.xVelocity(1)).*iTrack.precedingXVelocity(precedingId_position);
        temp.maxprecedingXVelocity = max(temp.precedingXVelocity);
        temp.minprecedingXVelocity = min(temp.precedingXVelocity);
        temp.meanprecedingXVelocity = mean(temp.precedingXVelocity);
        temp.stdprecedingXVelocity = std(temp.precedingXVelocity);
        
        % 相对速度信息,前车 - 自车
        temp.rVelocity = temp.precedingXVelocity - temp.xVelocity;
        temp.maxrVelocity = max(temp.rVelocity);
        temp.minrVelocity = min(temp.rVelocity);
        temp.meanrVelocity = mean(temp.rVelocity);
        temp.stdrVelocity = std(temp.rVelocity);
        
        % 自车加速度信息
        temp.xAcceleration = sign(iTrack.xVelocity(1)).*iTrack.xAcceleration(precedingId_position);
        temp.maxxAcceleration = max(temp.xAcceleration);
        temp.minxAcceleration = min(temp.xAcceleration);
        temp.meanxAcceleration = mean(temp.xAcceleration);
        temp.stdxAcceleration = std(temp.xAcceleration);
        
        % 跟车距离信息
        temp.dhw = iTrack.dhw(precedingId_position);
        temp.minDHW = min(temp.dhw);
        temp.maxDHW = max(temp.dhw);
        temp.meanDHW = mean(temp.dhw);
        temp.stdDHW = std(temp.dhw);
        
        % THW信息
        temp.thw = iTrack.thw(precedingId_position);
        temp.minTHW = min(temp.thw);
        
        % TTC存储
        temp.ttc = iTrack.ttc(precedingId_position);
        temp.minTTC = min(temp.ttc);
        
        % 其他信息
        temp.class = iTrack.class;
        temp.numLaneChanges = iTrack.numLaneChanges;
    end
%% -----------helper function-----------%
    function [kesai,DRAC] = SafetyIndicator(dhw,ttc,vh,vf)
        % 参数定义
        td = 0.2;
        row = 0.496;
        amax_accel = 3.084;
        amin_brake = 3.482;
        amax_brake = 5.688;
        DRAC = zeros(length(ttc), 1);
        brakedist = zeros(length(ttc), 1);
        % 计算制动距离
        for j = 1:length(ttc)
            if vh(j) > vf(j)
                brakedist(j) = (vh(j) - vf(j))*td+(vh(j)^2 - vf(j)^2)/(2*8);
            else
                brakedist(j) = (vh(j) - 0)*td+(vh(j)^2)/(2*8);
            end
        end
        
        % 计算安全距离
        safedist = vh.*row + 0.5*amax_accel*row.^2+(vh + row*amax_accel).^2/(2*amin_brake) - vf.^2./(2*amax_brake);
        
        % 计算kesai
        kesai = (dhw - brakedist)./(safedist - brakedist);
        
        % 计算DRAC-1
        for j = 1:length(DRAC)
            if vh(j) > vf(j)
                DRAC(j) = ttc(j)/(vh(j) - vf(j));
            else
                DRAC(j) = 10;
            end
        end
    end
%% 对初筛的数据再次处理
index = []; % 用于删除的index记录值
for iRow  = 1:length(tracks_temp3)
    iTrack = tracks_temp3(iRow);% 当前查找到的记录数据事件
    precedingID = iTrack.precedingId; % 跟驰的前车ID
    % 如果自车或跟随ID车辆存在换道行为，删除
    if (iTrack.numLaneChanges ~= 0) || (tracks(precedingID).numLaneChanges~= 0)
        index = [index;iRow];
        % TTC < 0，都是前车速度高于本车的安全行为
    elseif (iTrack.minTTC < 0) || (iTrack.minTHW < 0)
        index = [index;iRow];
        % 删除货车类型的数据
    elseif strcmp(iTrack.class,'Truck') || strcmp(tracks(precedingID).class,'Truck')
        index = [index;iRow];
        % 连续超过25个有效记录值，为一个合理的跟随事件，否则不认为处于稳定跟车状态，删除
    elseif length(iTrack.frames) < 25
        index = [index;iRow];
    end
end
tracks_temp3(index) = [];
Result = tracks_temp3;
%% 找到需要显示的index,只是用来展示，不影响Result的筛选值
index = [];
for i = 1:length(Result)
    temp = Result(i);
    index = [index;temp.id;temp.precedingId];
end
unique_iindex = unique(index);
tracks_filter = tracks(unique_iindex);
end

