class Change_detect:

    def __init__(self, M, eps, h):
        self.M = M              #window size
        self.eps = eps
        self.h = h              #threshold
        self.t = 0
        self.e_mean = 0
        self.cd_plus = 0         #cumulative positive deviation from mean
        self.cd_minus = 0        #cumulative negative deviation from mean

    def update(self,sample):
        self.t += 1
        if self.t <= self.M:
            self.e_mean += sample/self.M
            return 0
        else:                                                   #once window finished check of change detected
            d_plus = (sample - self.e_mean) - self.eps       #positive deviation from mean
            d_minus = -(sample - self.e_mean) - self.eps     #negative deviation from mean
            self.cd_plus = max(0, self.cd_plus + d_plus)
            self.cd_minus = max(0, self.cd_minus + d_minus)
            return self.cd_plus > self.h or self.cd_minus > self.h          #return true if change detected

    def reset(self):
        self.t = 0
        self.e_mean = 0
        self.cd_minus = 0
        self.cd_plus = 0