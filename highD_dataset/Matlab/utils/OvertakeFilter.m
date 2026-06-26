function [overtakefilter,Result] = OvertakeFilter(tracks)
%% 找到发生换道的数据帧
Result = {}; % 返回的是1*n的结构体
% Initialize tracks 
for iRow = 1:length(tracks)
% 	disp(iRow); % 调试用
    Temp = {}; % 返回的是1*n的结构体
    if (tracks(iRow).numLaneChanges == 1) && (tracks(iRow).class == "Car") % 说明找到了发生换道的数据帧
        itrack = tracks(iRow); 
        start_lane = itrack.lane(1); % 起始车道ID
        index = find(itrack.lane ~= start_lane,1); % highD记录发生换道点的index
        target_lane = itrack.lane(index); % 目标车道ID
        preced_frame_index = index - 100; % 向前推导50个帧，2s
        % 如果发现第一帧小于等于0，则从第一帧开始索引
        if preced_frame_index <= 0
            preced_frame_index = 1;
        end
        start_frame_index = find(abs(itrack.yVelocity(preced_frame_index:index))>= 0.05,1) + (preced_frame_index - 1); % 换道起始帧的索引
        end_frame_index = find(abs(itrack.yVelocity(index+1:end))<= 0.05,1) + index; % 换道结束帧的索引
        if isempty(end_frame_index) % 理论上跟车的换道终点超出记录
            end_frame_index = find(itrack.frames == itrack.frames(end)) ;
        end
        start_frame = itrack.frames(start_frame_index);% 换道开始的帧
        end_frame = itrack.frames(end_frame_index);% 换道结束的帧
        % 找到目标车道上的后车
        leftFollowingId_index = find(itrack.leftFollowingId(start_frame_index:end_frame_index)~=0,1) + (start_frame_index - 1 );
        rightFollowingId_index = find(itrack.rightFollowingId(start_frame_index:end_frame_index)~=0,1) + (start_frame_index - 1 );
        leftFollowingId = itrack.leftFollowingId(leftFollowingId_index);
        rightFollowingId = itrack.rightFollowingId(rightFollowingId_index);  
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
        FollowingId_start_index = find(tracks(FollowingId).frames == start_frame); % 目标车道后车的起始索引
        FollowingId_end_index = find(tracks(FollowingId).frames == end_frame); % 目标车道后车的终点索引
        if isempty(FollowingId_start_index) % 理论上跟车的换道起始点尚未被记录
            FollowingId_start_index = 1;
        end
        if isempty(FollowingId_end_index ) % 理论上换道终点,应该不会出现
            FollowingId_end_index = find(tracks(FollowingId).frames == tracks(FollowingId).frames(end));
        end
        % 根据初始纵向距离，判断属于cut-in还是overtake
        InitDhw = tracks(FollowingId).frontSightDistance(FollowingId_start_index) - itrack.frontSightDistance(start_frame_index); % 换道初始两车距离
        dx = abs(itrack.frontSightDistance(end_frame_index) - itrack.frontSightDistance(start_frame_index)); 
        if (InitDhw < 0) && (dx <=300)
        % 计算在换道起始时间内的切入特征  
            Temp.id = itrack.id; % 提取车辆的ID号码
            Temp.FollowingId = FollowingId; % 提取跟车的ID号码
            Temp.xVelocity = abs(itrack.xVelocity(start_frame_index:end_frame_index));
            Temp.yVelocity = itrack.yVelocity(start_frame_index:end_frame_index);
            Temp.xAcceleration = itrack.xAcceleration(start_frame_index:end_frame_index).*sign(itrack.xVelocity(start_frame_index));
            Temp.yAcceleration = itrack.yAcceleration(start_frame_index:end_frame_index).*sign(Temp.yVelocity(1));
            Temp.duration = (itrack.frames(end_frame_index) - itrack.frames(start_frame_index)) * 0.04;
            Temp.dx = abs(itrack.frontSightDistance(end_frame_index) - itrack.frontSightDistance(start_frame_index)); % 换道过程行驶的纵向距离
            Temp.InitDhw = tracks(FollowingId).frontSightDistance(FollowingId_start_index) - itrack.frontSightDistance(start_frame_index) - 5; % 换道初始两车距离
            Temp.InitFollowingXVelocity = abs(tracks(FollowingId).xVelocity(FollowingId_start_index)); % 换道初始后车速度
            Temp.FollowingXVelocity = tracks(FollowingId).xVelocity(FollowingId_start_index:FollowingId_end_index).*sign(tracks(FollowingId).xVelocity(FollowingId_start_index)); % 换道过程中后车速度
            Temp.InitFollowingXAcce = tracks(FollowingId).xAcceleration(FollowingId_start_index).*sign(tracks(FollowingId).xVelocity(FollowingId_start_index)); % 换道初始后车加速度
            Temp.FollowingXAcce = tracks(FollowingId).xAcceleration(FollowingId_start_index:FollowingId_end_index).*sign(tracks(FollowingId).xVelocity(FollowingId_start_index)); % 换道过程中后车加速度
        else
            continue; 
        end
	else
        continue; % 该条数据没有发生换道
    end
    % 堆叠数据
    Result = [Result;Temp];
end
iindex = [];
for i = 1:size(Result,1)
    temp = cell2mat(Result(i));
	iindex = [iindex;temp.id;temp.FollowingId];
end
% 找到唯一的索引号
unique_iindex = unique(iindex);
overtakefilter = tracks(unique_iindex);
end