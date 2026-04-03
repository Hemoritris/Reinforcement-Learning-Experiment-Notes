##################迭代策略评估算法实现#########################
import numpy as np
import copy
#矩阵扩维和降维度
#print("P_pi",Pi[1,:],np.expand_dims(Pi[1,:],axis=0))
#print(np.dot(np.expand_dims(Pi[1,:],axis=0),P_ssa[:,1,:]).squeeze())
class Maze:
    def __init__(self):
        #初始化行为值函数
        self.qvalue = -0.01*np.zeros((16,4))
        #初始化每个状态-动作对的次数
        self.n = 0*np.ones((16,4))
        self.states = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
        self.actions = np.array([0,1,2,3])
        self.gamma = 1.0
        #初始化策略
        self.Pi = 0.25 * np.ones((16, 4))
        self.cur_state = 0
        self.cur_action = 0
        self.old_policy=np.ones((16,4))
        #################1.状态转移概率P(s'|s,a)模型构建#################################
        self.P_ssa = np.zeros((4, 16, 16))
        # print(P_ssa)
        P1 = np.zeros((16, 16))
        P2 = np.zeros((16, 16))
        P3 = np.zeros((16, 16))
        P4 = np.zeros((16, 16))
        # 状态1处的转移
        P1[1, 2] = 1
        P2[1, 5] = 1
        P3[1, 0] = 1
        P4[1, 1] = 1
        # 状态2处的转移
        P1[2, 3] = 1
        P2[2, 6] = 1
        P3[2, 1] = 1
        P4[2, 2] = 1
        # 状态3处的转移
        P1[3, 3] = 1
        P2[3, 7] = 1
        P3[3, 2] = 1
        P4[3, 3] = 1
        # 状态4处的转移
        P1[4, 5] = 1
        P2[4, 8] = 1
        P3[4, 4] = 1
        P4[4, 0] = 1
        # 状态5处的转移
        P1[5, 6] = 1
        P2[5, 9] = 1
        P3[5, 4] = 1
        P4[5, 1] = 1
        # 状态6处的转移
        P1[6, 7] = 1
        P2[6, 10] = 1
        P3[6, 5] = 1
        P4[6, 2] = 1
        # 状态7处的转移
        P1[7, 7] = 1
        P2[7, 11] = 1
        P3[7, 6] = 1
        P4[7, 3] = 1
        # 状态8处的转移
        P1[8, 9] = 1
        P2[8, 12] = 1
        P3[8, 8] = 1
        P4[8, 4] = 1
        # 状态9处的转移
        P1[9, 10] = 1
        P2[9, 13] = 1
        P3[9, 8] = 1
        P4[9, 5] = 1
        # 状态10处的转移
        P1[10, 11] = 1
        P2[10, 14] = 1
        P3[10, 9] = 1
        P4[10, 6] = 1
        # 状态11处的转移
        P1[11, 11] = 1
        P2[11, 15] = 1
        P3[11, 10] = 1
        P4[11, 7] = 1
        # 状态12处的转移
        P1[12, 13] = 1
        P2[12, 12] = 1
        P3[12, 12] = 1
        P4[12, 8] = 1
        # 状态13处的转移
        P1[13, 14] = 1
        P2[13, 13] = 1
        P3[13, 12] = 1
        P4[13, 9] = 1
        # 状态14处的转移
        P1[14, 15] = 1
        P2[14, 14] = 1
        P3[14, 13] = 1
        P4[14, 10] = 1
        self.P_ssa[0, :, :] = P1
        self.P_ssa[1, :, :] = P2
        self.P_ssa[2, :, :] = P3
        self.P_ssa[3, :, :] = P4
        ###############2.回报模型构建########################
        self.r_sa = -np.ones((16, 4))
        self.r_sa[0, :] = 0
        self.r_sa[15, :] = 0
        # self.r_sa[1,2]=1
    #重置环境函数
    def reset(self):
        # 初始化行为值函数
        self.qvalue = -0.01*np.zeros((16, 4))
        # 初始化每个状态-动作对的次数
        self.n = 0* np.ones((16, 4))
    #探索初始化函数
    def explore_init(self):
        state_prob = (1/16)*np.array([1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1])
        s0 = np.random.choice(self.states,p=state_prob)
        a0 = np.random.choice(self.actions,p=[0.25,0.25,0.25,0.25])
        return s0,a0
    #根据策略pi采样一个动作
    def sample_action(self,state):
        action = np.random.choice(self.actions,p=self.Pi[state,:])
        return action
    #跟环境交互一步
    def step(self,action):
        flag = False
        trans_prob = self.P_ssa[action, self.cur_state]
        if(self.cur_state==0 or self.cur_state==15):
            next_state = self.cur_state
        else:
            next_state = np.random.choice(self.states,p=trans_prob)
        r_next = self.r_sa[self.cur_state,action]
        if next_state == 0 or next_state==15:
            flag = True
        #print(self.cur_state,action,next_state)
        return next_state,r_next,flag
    #############策略改进源代码##########
    def update_policy(self):
        for i in range(16):
            self.Pi[i,:]=0
            max_num = np.argmax(self.qvalue[i,:])
            self.Pi[i, max_num] = 1
    def update_epsilon_greedy(self):
        epsilon = 0.2
        for i in range(16):
            self.Pi[i,:]=epsilon/4
            max_num = np.argmax(self.qvalue[i,:])
            self.Pi[i, max_num] = 0.85


    #蒙特卡洛强化学习算法
    def MC_learning(self):
        num = 0
        while num<6000:
            num+=1
            flag=False
            #采样一条轨迹
            state_traj=[]
            action_traj = []
            reward_traj=[]
            g = 0
            episode_num = 0
            while flag==False and episode_num<200:
                #与环境交互一次
                # 探索初始化
                if episode_num==0:
                    cur_state, cur_action = self.explore_init()
                    self.cur_state = cur_state
                else:
                    cur_action = self.sample_action(self.cur_state)
                state_traj.append(self.cur_state)
                action_traj.append(cur_action)
                next_state, reward, flag = self.step(cur_action)
                reward_traj.append(reward)
                self.cur_state = next_state
                episode_num += 1

            # print("state_traj",state_traj)
            ############利用采集到的轨迹更新行为值函数################
            for i in reversed(range(len(state_traj))):
                #计算状态-动作对(s,a)的访问频次
                self.n[state_traj[i],action_traj[i]]+=1.0
                #利用增量式方式更新当前状态动作值
                g*=self.gamma
                g+=reward_traj[i]
                self.qvalue[state_traj[i],action_traj[i]]=\
                    (self.qvalue[state_traj[i], action_traj[i]]*(self.n[state_traj[i],action_traj[i]]-1)+g)/ \
                    self.n[state_traj[i], action_traj[i]]
            if state_traj[0] == 1 and action_traj[0] == 3:
                print("state_traj", state_traj)
                print("状态频次及值函数", self.n[1, 3],self.qvalue[1,3] )
            ###########更新策略################
            if num%501==0:
                self.old_policy = copy.deepcopy(self.Pi)
                self.update_policy()
                self.n = np.zeros((16, 4))
                delta = np.linalg.norm(self.old_policy - self.Pi)
                print("delta",delta)
                # self.reset()
            # print("策略",self.Pi)
            # print("old policy",old_policy)
        # self.update_policy()

def q_ana_evaluate(Pi,r_sa,P_ssa):
    P_pi = np.zeros((16, 16))
    C_pi = np.zeros((16, 1))
    for i in range(16):
        # 计算pi(a|s)*p(s'|s,a)
        P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
        # 计算pi(a|s)*r(s,a)
        C_pi[i, :] = np.dot(r_sa[i, :], Pi[i, :])
    ############解析法计算值函数######################
    M = np.eye(16) - P_pi
    I_M = np.linalg.inv(M)
    V = np.dot(I_M, C_pi)
    #计算行为值函数
    q_value = np.zeros((16, 4))
    for i in range(16):
        q_sa = np.zeros((1, 4))
        for j in range(4):
            Pi[i, :] = 0
            Pi[i, j] = 1
            P_pi[i, :] = np.dot(np.expand_dims(Pi[i, :], axis=0), P_ssa[:, i, :]).squeeze()
            vi = np.dot(r_sa[i, :], Pi[i, :]) + np.dot(P_pi[i, :], V.squeeze())
            q_sa[0, j] = vi
        q_value[i, :] = q_sa[0, :]
    return q_value



if __name__ == '__main__':
    ######实例化对象####
    maze=Maze()
    maze.reset()
    print("initial policy",maze.Pi)
    maze.MC_learning()
    print("Final policy",maze.Pi)
    print("optimal qvalue",maze.qvalue)
    q_real_value = q_ana_evaluate(maze.Pi,maze.r_sa,maze.P_ssa)
    print("real qvalue:",q_real_value)
    print("访问频次",maze.n)