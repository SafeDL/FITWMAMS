function [cutinfilter,Result] = CutInFilter(tracks)
%% 根据以下条件初筛前车切入事件：
% 与跟驰事件筛选不同的是，变换了视角，现在从本车换道影响后车来搜索Cut-in Scenario
% 1：遍历车辆ID，根据是否换道预筛选
% 2：找到换道发生点，找出换道全过程的index
% 3：查看全过程index的左右车道内是否有后车行驶
% 4：连续超过25个有效记录值，为一个合理的Cut In事件
%% 找到发生换道的数据帧
tracks_temp = []; % 返回的是1*n的结构体
% Initialize tracks 
for iRow = 1:length(tracks)
% 	disp(iRow); % 调试用
    temp = {}; % 返回的是1*n的结构体
    if (tracks(iRow).numLaneChanges == 1) && (tracks(iRow).class == "Car") % 说明找到了发生换道的数据帧
        iTrack = tracks(iRow);
        start_lane = iTrack.lane(1); % 起始车道ID
        index = find(iTrack.lane ~= start_lane,1); % highD记录发生换道点的index
        target_lane = iTrack.lane(index); % 目标车道ID
        preced_frame_index = index - 125; % 向前推导125个帧，5s
        % 如果发现第一帧小于等于0，则从第一帧开始索引
        if preced_frame_index <= 0
            preced_frame_index = 1;
        end
        start_frame_index = find(abs(iTrack.yVelocity(preced_frame_index:index))>= 0.05,1) + (preced_frame_index - 1); % iTrack换道起始帧索引
        end_frame_index = find(abs(iTrack.yVelocity(index+1:end))<= 0.05,1) + index; % iTrack换道结束帧索引
        if isempty(end_frame_index) % 理论上跟车的换道终点超出记录
            end_frame_index = find(iTrack.frames == iTrack.frames(end)) ;
        end
        start_frame = iTrack.frames(start_frame_index);% iTrack换道起始帧
        end_frame = iTrack.frames(end_frame_index);%  iTrack换道结束帧
        
        % 找到目标车道上的后车
        leftFollowingId_index = find(iTrack.leftFollowingId(start_frame_index:end_frame_index)~=0,1) + (start_frame_index - 1 );
        rightFollowingId_index = find(iTrack.rightFollowingId(start_frame_index:end_frame_index)~=0,1) + (start_frame_index - 1 );
        leftFollowingId = iTrack.leftFollowingId(leftFollowingId_index);
        rightFollowingId = iTrack.rightFollowingId(rightFollowingId_index);  
        
        if isempty(leftFollowingId) && isempty(rightFollowingId)
            continue; % 属于自由换道
        else
            if (~isempty(leftFollowingId)) && (tracks(leftFollowingId).lane(1) == target_lane) && (tracks(leftFollowingId).numLaneChanges ~= 1) && (tracks(leftFollowingId).class ~= "Truck") % 车辆向左换道
                FollowingId = leftFollowingId;
            elseif (~isempty(rightFollowingId)) && (tracks(rightFollowingId).lane(1) == target_lane) && (tracks(rightFollowingId).numLaneChanges ~= 1) && (tracks(rightFollowingId).class ~= "Truck") % 车辆向右换道
                FollowingId = rightFollowingId;
            else
                continue; % 自由换道
            end
        end
        
        FollowingId_start_index = find(tracks(FollowingId).frames == start_frame); % 目标车的起始索引
        FollowingId_end_index = find(tracks(FollowingId).frames == end_frame); % 目标车的终止索引
        % 根据iTrack换道发生点index找到跟随车的index ， 只有这段时间的TTC才有计算必要
        FollowingId_index = find(tracks(FollowingId).frames == iTrack.frames(index));
        
        % FollowingId_index为空，说明没有完整的记录换道发生时后车的轨迹，跳过
        if isempty(FollowingId_index)
            continue;
        end
        
        if isempty(FollowingId_start_index) % 理论上跟车的换道起始点尚未被记录
            FollowingId_start_index = 1;
            start_frame_index = find(iTrack.frames == tracks(FollowingId).frames(1)); % 初始帧对齐
            start_frame = iTrack.frames(start_frame_index);% 换道开始的帧
        end
        if isempty(FollowingId_end_index ) % 换道终点超出记录
            FollowingId_end_index = find(tracks(FollowingId).frames == tracks(FollowingId).frames(end));
            end_frame_index = find(iTrack.frames == tracks(FollowingId).frames(FollowingId_end_index)) ; % 结束帧对齐
            end_frame = iTrack.frames(end_frame_index);% 换道结束的帧
        end
        % 数据校核,如果发现dhw存在负值，说明该段数据作废
        dhw = tracks(FollowingId).frontSightDistance(FollowingId_start_index:FollowingId_end_index) - iTrack.frontSightDistance(start_frame_index:end_frame_index) - 4.75;
        if ~isempty(find(dhw < 0, 1))
            continue;
        end
        % 根据初始纵向距离，判断属于cut-in还是overtake
        Initdhw = tracks(FollowingId).frontSightDistance(FollowingId_start_index) - iTrack.frontSightDistance(start_frame_index) - 4.75; % 换道初始两车距离
        time = (end_frame_index - start_frame_index)*0.04; % 换道时间
        dx = abs(iTrack.frontSightDistance(end_frame_index) - iTrack.frontSightDistance(start_frame_index)); % 换道车纵向行驶距离
        if (Initdhw > 0) && (Initdhw <= 150) && (dx <= 300)
        % 计算在换道起始时间内的切入特征  
            temp.id = iTrack.id; % 提取换道车辆的ID号码
            temp.FollowingId = FollowingId; % 提取跟车的ID号码
            % 切入车的纵向速度
            temp.xVelocity = abs(iTrack.xVelocity(start_frame_index:end_frame_index));
            temp.maxVelocity = max(temp.xVelocity);
            temp.minVelocity = min(temp.xVelocity);
            temp.meanVelocity = mean(temp.xVelocity);
            temp.stdVelocity = std(temp.xVelocity);
            
            % 后车速度信息
            temp.FollowingXVelocity = abs(tracks(FollowingId).xVelocity(FollowingId_start_index:FollowingId_end_index));
            temp.maxFollowingXVelocity = max(temp.FollowingXVelocity);
            temp.minFollowingXVelocity = min(temp.FollowingXVelocity);
            temp.meanFollowingXVelocity = mean(temp.FollowingXVelocity);
            temp.stdFollowingXVelocity = std(temp.FollowingXVelocity);
            
            % 相对速度信息,前车 - 自车
            temp.rVelocity = temp.xVelocity - temp.FollowingXVelocity;
            temp.maxrVelocity = max(temp.rVelocity);
            temp.minrVelocity = min(temp.rVelocity);
            temp.meanrVelocity = mean(temp.rVelocity);
            temp.stdrVelocity = std(temp.rVelocity);
            
            % 切入车加速度信息
            temp.xAcceleration = iTrack.xAcceleration(start_frame_index:end_frame_index).*sign(iTrack.xVelocity(start_frame_index));
            temp.maxxAcceleration = max(temp.xAcceleration);
            temp.minxAcceleration = min(temp.xAcceleration);
            temp.meanxAcceleration = mean(temp.xAcceleration);
            temp.stdxAcceleration = std(temp.xAcceleration);
            
            % 后车加速度信息
            temp.FollowingxAcceleration = tracks(FollowingId).xAcceleration(FollowingId_start_index:FollowingId_end_index).*sign(tracks(FollowingId).xVelocity(FollowingId_start_index));
            temp.maxFollowingxAcceleration = max(temp.FollowingxAcceleration);
            temp.minFollowingxAcceleration = min(temp.FollowingxAcceleration);
            temp.meanFollowingxAcceleration = mean(temp.FollowingxAcceleration);
            temp.stdFollowingxAcceleration = std(temp.FollowingxAcceleration);
                
            % 跟车距离信息
            temp.dhw = dhw;
            temp.minDHW = min(temp.dhw);
            temp.maxDHW = max(temp.dhw);
            temp.meanDHW = mean(temp.dhw);
            temp.stdDHW = std(temp.dhw);
                    
            % 后车与前车相距的TTC THW kesai DRAC等安全指标计算
            temp.ttc = tracks(FollowingId).ttc(FollowingId_index:FollowingId_end_index);
            temp.thw = tracks(FollowingId).thw(FollowingId_index:FollowingId_end_index);
            dhw_useful = tracks(FollowingId).dhw(FollowingId_index:FollowingId_end_index); 
            vh = abs(tracks(FollowingId).xVelocity(FollowingId_index:FollowingId_end_index));
            vf = abs(iTrack.xVelocity(index:end_frame_index));
                       
            [kesai,DRAC] = SafetyIndicator(dhw_useful, temp.ttc ,vh, vf);
            temp.kesai = kesai;
            temp.DRAC = DRAC;
            
            % 换道时间及初始距离，加速度等
            temp.duration = time;
            temp.dx = abs(iTrack.frontSightDistance(end_frame_index) - iTrack.frontSightDistance(start_frame_index)); % 换道车纵向行驶距离
            temp.Initdhw = Initdhw;
            temp.InitXAcce = tracks(FollowingId).xAcceleration(FollowingId_start_index).*sign(tracks(FollowingId).xVelocity(FollowingId_start_index)); % 换道初始后车加速度
        else
            continue; % Initdhw < 0,属于超车；Initdhw>150，无意义;dx > 300，无意义
        end
	else
        continue; % 该条数据没有发生换道
    end
    % 堆叠数据
    tracks_temp = [tracks_temp;temp];
end
%% 对初筛的数据再次处理
index = []; % 用于删除的index记录值
for iRow  = 1:length(tracks_temp)
    iTrack = tracks_temp(iRow); % 当前查找到的记录数据事件
    if length(iTrack.xVelocity) < 25
        index = [index;iRow];
    % TTC < 0，都是前车速度高于本车的安全行为
    elseif (min(iTrack.ttc) < 0) || (min(iTrack.thw) < 0)
        index = [index;iRow];
    end
end
tracks_temp(index) = [];
Result = tracks_temp;

index = [];
for iRow = 1:size(Result,1)
    temp = Result(iRow);
	index = [index;temp.id;temp.FollowingId];
end
% 找到唯一的索引号
unique_iindex = unique(index);
cutinfilter = tracks(unique_iindex);
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
        for i = 1:length(ttc)
            if vh(i) > vf(i)
                brakedist(i) = (vh(i) - vf(i))*td+(vh(i)^2 - vf(i)^2)/(2*8);
            else
                brakedist(i) = (vh(i) - 0)*td+(vh(i)^2)/(2*8);
            end
        end
       
        % 计算安全距离
        safedist = vh.*row + 0.5*amax_accel*row.^2+(vh + row*amax_accel).^2/(2*amin_brake) - vf.^2./(2*amax_brake);
        
        % 计算kesai
        kesai = (dhw - brakedist)./(safedist - brakedist);
        
        % 计算DRAC-1
        for i = 1:length(DRAC)
            if vh(i) > vf(i)
                DRAC(i) = ttc(i)/(vh(i) - vf(i));
            else
                DRAC(i) = 10;
            end
        end
    end

end