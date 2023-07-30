class Lane:
    def __init__(self, name):
        self.vehicle_num = 0  # 车辆数目
        self.track_num = 0  # 追踪数目, 等于车辆数目
        self.max_wait = 0  # 最大等待时间， 也可以理解为首车等待时间
        self.emergy_num = 0  # 紧急车辆数目
        self.all_wait = 0  # 所有车加起来的等待时间
        self.emergy_wait = 0
        self.stop_time = 0
        self.name = name
        self.emerge_max_wait = 0
        self.emergy_num_0 = 0  # 紧急车辆数目
        self.emergy_wait_0 = 0
        self.emerge_max_wait_0 = 0


