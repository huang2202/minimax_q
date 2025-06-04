import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import pickle

class MinimaxQLearning:
    """Minimax Q-learning算法实现"""
    
    def __init__(self, alpha=0.1, gamma=0.9, epsilon=0.1):
        """
        参数:
        alpha: 学习率
        gamma: 折扣因子
        epsilon: 探索率
        """
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))  # Q表
        self.V = defaultdict(float)  # 状态值函数
        
    def get_state_key(self, state):
        """将状态转换为可哈希的键"""
        return tuple(map(tuple, state))
    
    def get_available_actions(self, state):
        """获取当前状态下的可用动作"""
        actions = []
        for i in range(len(state)):
            for j in range(len(state[0])):
                if state[i][j] == 0:
                    actions.append((i, j))
        return actions
    
    def choose_action(self, state, available_actions):
        """使用epsilon-greedy策略选择动作"""
        if random.random() < self.epsilon:
            return random.choice(available_actions)
        else:
            state_key = self.get_state_key(state)
            q_values = {action: self.Q[state_key][action] 
                       for action in available_actions}
            max_q = max(q_values.values())
            best_actions = [a for a, q in q_values.items() if q == max_q]
            return random.choice(best_actions)
    
    def update_minimax_value(self, state, player):
        """更新状态的minimax值"""
        state_key = self.get_state_key(state)
        available_actions = self.get_available_actions(state)
        
        if not available_actions:  # 终止状态
            return
        
        if player == 1:  # Max player
            self.V[state_key] = max(self.Q[state_key][a] 
                                   for a in available_actions)
        else:  # Min player
            self.V[state_key] = min(self.Q[state_key][a] 
                                   for a in available_actions)
    
    def update_q_value(self, state, action, next_state, reward, done, player):
        """更新Q值"""
        state_key = self.get_state_key(state)
        next_state_key = self.get_state_key(next_state)
        
        if done:
            target = reward
        else:
            target = reward + self.gamma * self.V[next_state_key]
        
        # Q-learning更新规则
        self.Q[state_key][action] += self.alpha * (
            target - self.Q[state_key][action]
        )
        
        # 更新状态值
        self.update_minimax_value(state, player)
    
    def decay_epsilon(self, decay_rate=0.995):
        """衰减探索率"""
        self.epsilon *= decay_rate
        self.epsilon = max(0.01, self.epsilon)  # 保持最小探索率

class GridGame:
    """5x5网格游戏环境"""
    
    def __init__(self, size=5):
        self.size = size
        self.reset()
        
    def reset(self):
        """重置游戏"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.current_player = 1
        return self.board.copy()
    
    def step(self, action):
        """执行动作"""
        i, j = action
        if self.board[i][j] != 0:
            return self.board.copy(), -10, True  # 非法动作
        
        self.board[i][j] = self.current_player
        
        # 检查获胜
        winner = self.check_winner()
        if winner != 0:
            reward = 1 if winner == 1 else -1
            return self.board.copy(), reward, True
        
        # 检查平局
        if np.all(self.board != 0):
            return self.board.copy(), 0, True
        
        # 切换玩家
        self.current_player = -self.current_player
        return self.board.copy(), 0, False
    
    def check_winner(self):
        """检查获胜者（简化版：检查行、列、对角线的4连）"""
        # 检查行
        for i in range(self.size):
            for j in range(self.size - 3):
                if self.board[i][j] != 0 and \
                   all(self.board[i][j] == self.board[i][j+k] for k in range(4)):
                    return self.board[i][j]
        
        # 检查列
        for i in range(self.size - 3):
            for j in range(self.size):
                if self.board[i][j] != 0 and \
                   all(self.board[i][j] == self.board[i+k][j] for k in range(4)):
                    return self.board[i][j]
        
        # 检查对角线
        for i in range(self.size - 3):
            for j in range(self.size - 3):
                if self.board[i][j] != 0 and \
                   all(self.board[i][j] == self.board[i+k][j+k] for k in range(4)):
                    return self.board[i][j]
                
                if self.board[i][j+3] != 0 and \
                   all(self.board[i][j+3] == self.board[i+k][j+3-k] for k in range(4)):
                    return self.board[i][j+3]
        
        return 0

def train_agents(episodes=10000):
    """训练两个智能体"""
    env = GridGame()
    agent1 = MinimaxQLearning(alpha=0.1, gamma=0.9, epsilon=0.3)
    agent2 = MinimaxQLearning(alpha=0.1, gamma=0.9, epsilon=0.3)
    
    win_rates = []
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_rewards = []
        done = False
        
        # 存储轨迹
        trajectory = []
        
        while not done:
            current_player = env.current_player
            agent = agent1 if current_player == 1 else agent2
            
            # 选择动作
            available_actions = agent.get_available_actions(state)
            if not available_actions:
                break
                
            action = agent.choose_action(state, available_actions)
            
            # 执行动作
            next_state, reward, done = env.step(action)
            
            # 存储经验
            trajectory.append((state.copy(), action, reward, next_state.copy(), done, current_player))
            
            episode_rewards.append(reward)
            state = next_state
        
        # 更新Q值（从轨迹末尾开始）
        for i in range(len(trajectory) - 1, -1, -1):
            s, a, r, ns, d, p = trajectory[i]
            if p == 1:
                agent1.update_q_value(s, a, ns, r, d, p)
            else:
                agent2.update_q_value(s, a, ns, -r, d, p)  # 注意对手的奖励是负的
        
        # 衰减探索率
        if episode % 100 == 0:
            agent1.decay_epsilon()
            agent2.decay_epsilon()
        
        # 记录统计信息
        if episode % 100 == 0:
            print(f"Episode {episode}, Epsilon: {agent1.epsilon:.3f}")
            
        rewards_history.append(sum(episode_rewards))
    
    return agent1, agent2, rewards_history

def evaluate_agents(agent1, agent2, games=100):
    """评估训练后的智能体"""
    env = GridGame()
    results = {'player1_wins': 0, 'player2_wins': 0, 'draws': 0}
    
    # 保存原始epsilon
    eps1, eps2 = agent1.epsilon, agent2.epsilon
    agent1.epsilon = 0  # 评估时不探索
    agent2.epsilon = 0
    
    for _ in range(games):
        state = env.reset()
        done = False
        
        while not done:
            current_player = env.current_player
            agent = agent1 if current_player == 1 else agent2
            
            available_actions = agent.get_available_actions(state)
            if not available_actions:
                break
                
            action = agent.choose_action(state, available_actions)
            state, reward, done = env.step(action)
            
            if done:
                if reward > 0:
                    results['player1_wins'] += 1
                elif reward < 0:
                    results['player2_wins'] += 1
                else:
                    results['draws'] += 1
    
    # 恢复epsilon
    agent1.epsilon = eps1
    agent2.epsilon = eps2
    
    return results

def visualize_results(rewards_history):
    """可视化训练结果"""
    plt.figure(figsize=(12, 5))
    
    # 奖励曲线
    plt.subplot(1, 2, 1)
    window = 100
    avg_rewards = [np.mean(rewards_history[i:i+window]) 
                   for i in range(0, len(rewards_history), window)]
    plt.plot(avg_rewards)
    plt.title('Average Rewards per 100 Episodes')
    plt.xlabel('Episodes (x100)')
    plt.ylabel('Average Reward')
    
    # Q值热力图示例
    plt.subplot(1, 2, 2)
    # 这里可以可视化某个状态的Q值
    plt.title('Q-values Heatmap (Example)')
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

def save_agents(agent1, agent2, filename='trained_agents.pkl'):
    """保存训练好的智能体"""
    with open(filename, 'wb') as f:
        pickle.dump({'agent1': agent1, 'agent2': agent2}, f)

def load_agents(filename='trained_agents.pkl'):
    """加载训练好的智能体"""
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data['agent1'], data['agent2']

# 主训练流程
if __name__ == "__main__":
    print("开始训练Minimax Q-learning智能体...")
    
    # 训练
    agent1, agent2, rewards = train_agents(episodes=5000)
    
    # 评估
    print("\n评估智能体性能...")
    results = evaluate_agents(agent1, agent2, games=100)
    print(f"Player 1 胜率: {results['player1_wins']}%")
    print(f"Player 2 胜率: {results['player2_wins']}%")
    print(f"平局率: {results['draws']}%")
    
    # 保存模型
    save_agents(agent1, agent2)
    
    # 可视化
    visualize_results(rewards)